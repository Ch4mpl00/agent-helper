import OpenAI from "openai";
import { Duration, Effect, Schedule } from "effect";
import { toError } from "../errors";
import type { ChatProvider, CompletionParams, CompletionResult } from "./types";

// Transient-failure retry, factored OUT of individual providers. Retrying is
// a cross-cutting reliability policy, not a property of one endpoint: the
// workflow compiler is on the hot path no matter which provider its model
// routes to (AGENT_COMPILER_MODEL switches the route silently), so the
// engine wraps EVERY provider with `withRetry` at startup. 408/429 and
// 5xx (overload / transient server error) back off and retry; any other
// 4xx is permanent and rethrows immediately.
//
// Visibility contract: a retry must never look like one slow call. When the
// caller passes `CompletionParams.trace`, every retry attempt emits a
// WARNING `llm_retry` event on that scope — attempt number, HTTP status and
// backoff delay land in the Langfuse trace right next to the generation
// they delayed. Without a trace (scripts), retries still go to stderr.

export interface RetryInfo {
  // 1-based number of the attempt that just FAILED.
  attempt: number;
  // HTTP status of the failure, when the error was an APIError.
  status?: number;
  delayMs: number;
}

export interface RetryOpts {
  // Retries after the initial attempt. Default 4, with exponential backoff.
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  jitter?: boolean;
}

function isTransient(error: Error): boolean {
  return error instanceof OpenAI.APIConnectionError ||
    (error instanceof OpenAI.APIError && error.status !== undefined &&
      (error.status === 408 || error.status === 429 || error.status >= 500));
}

// Effect owns the retry clock and cancellation, including the backoff sleep.
// Concrete providers disable SDK retries so every attempt is visible here.
export function retryOnTransientEffect<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  opts: RetryOpts & { onRetry?: (info: RetryInfo) => void } = {},
): Effect.Effect<T, Error> {
  const exponential = Schedule.exponential(opts.baseDelayMs ?? 2000);
  const schedule = (opts.jitter === false ? exponential : Schedule.jittered(exponential)).pipe(
    Schedule.modifyDelay(({ duration }) => Effect.succeed(Math.min(Duration.toMillis(duration), opts.maxDelayMs ?? 30_000))),
    Schedule.tap(({ attempt, input, duration }) => Effect.sync(() => {
      // Schedule decisions are evaluated before retry's times/while guards.
      if (!(input instanceof Error) || !isTransient(input) || attempt > (opts.maxRetries ?? 4)) return;
      opts.onRetry?.({
        attempt,
        status: input instanceof OpenAI.APIError ? input.status : undefined,
        delayMs: Duration.toMillis(duration),
      });
    })),
  );
  return Effect.tryPromise({ try: fn, catch: toError }).pipe(
    Effect.retry({ schedule, times: opts.maxRetries ?? 4, while: isTransient }),
  );
}

// Promise boundary for scripts and the existing ChatProvider interface.
export function retryOnTransient<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  opts: RetryOpts & { signal?: AbortSignal; onRetry?: (info: RetryInfo) => void } = {},
): Promise<T> {
  return Effect.runPromise(retryOnTransientEffect(fn, opts), { signal: opts.signal });
}

export function withRetry(provider: ChatProvider, opts: RetryOpts = {}): ChatProvider {
  return {
    kind: provider.kind,
    complete(params: CompletionParams): Promise<CompletionResult> {
      return retryOnTransient((signal) => provider.complete({ ...params, signal }), {
        ...opts,
        signal: params.signal,
        onRetry: ({ attempt, status, delayMs }) => {
          console.warn(
            `[retry] ${provider.kind}/${params.model} attempt ${attempt} failed` +
              ` (status=${status ?? "?"}), retrying in ${delayMs}ms`,
          );
          params.trace?.event({
            name: "llm_retry",
            level: "WARNING",
            metadata: {
              provider: provider.kind,
              model: params.model,
              attempt,
              status: status ?? null,
              delay_ms: delayMs,
            },
          });
        },
      });
    },
  };
}
