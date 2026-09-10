import { Cause, Effect, Exit } from "effect";
import type { ChatProvider, CompletionParams, CompletionResult } from "./providers";
import type { GenerationEndOpts, GenerationStartOpts, TraceContext } from "./tracing";
import { toError } from "./errors";

// One budget covers the request and every retry/backoff, rather than restarting
// the timeout on each attempt. Callers can choose a shorter deadline.
export const DEFAULT_GENERATION_TIMEOUT_MS = 600_000;

export interface GenerationOpts {
  provider: ChatProvider;
  params: CompletionParams;
  scope: TraceContext;
  observation: Omit<GenerationStartOpts, "model" | "input">;
  timeoutMs?: number;
  describe?: (result: CompletionResult) => GenerationEndOpts;
}

export function generationEffect(opts: GenerationOpts): Effect.Effect<CompletionResult, Error> {
  return Effect.suspend(() => traceGenerationEffect({
    scope: opts.scope,
    observation: { ...opts.observation, model: opts.params.model, input: structuredClone(opts.params.messages) },
    timeoutMs: opts.timeoutMs,
    run: (signal) => opts.provider.complete({ ...opts.params, trace: opts.scope, signal }),
    describe: opts.describe ?? ((result) => ({ output: result.message, usage: result.usage })),
  }));
}

// The compiler validates its result before closing the generation, since only
// an accepted plan gets the planner tag used by the judge. Keep that policy at
// the call site while sharing deadline and finalization with the other callers.
export function traceGenerationEffect<A>(opts: {
  scope: TraceContext;
  observation: GenerationStartOpts;
  run: (signal: AbortSignal) => Promise<A>;
  describe: (value: A) => GenerationEndOpts;
  timeoutMs?: number;
}): Effect.Effect<A, Error> {
  return Effect.suspend(() => {
    const timeoutMs = opts.timeoutMs ?? DEFAULT_GENERATION_TIMEOUT_MS;
    const generation = opts.scope.generation(opts.observation);
    return Effect.tryPromise({
      try: opts.run,
      catch: toError,
    }).pipe(
      Effect.timeoutOrElse({
        duration: timeoutMs,
        orElse: () => Effect.fail(new Error(`Generation timed out after ${timeoutMs}ms`)),
      }),
      Effect.onExit((exit) => Effect.sync(() => {
        if (Exit.isSuccess(exit)) {
          generation.end(opts.describe(exit.value));
        } else {
          const error = toError(Cause.squash(exit.cause));
          generation.end({ output: { error: error.message }, level: "ERROR", statusMessage: error.message });
        }
      })),
    );
  });
}

export function runGeneration(opts: GenerationOpts): Promise<CompletionResult> {
  return Effect.runPromise(generationEffect(opts), { signal: opts.params.signal });
}
