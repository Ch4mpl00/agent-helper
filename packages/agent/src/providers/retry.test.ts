import { Effect, Fiber } from "effect";
import { TestClock } from "effect/testing";
import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";
import { runGeneration } from "../generation";
import { nullTracer } from "../tracing";
import { retryOnTransientEffect, withRetry, type RetryInfo } from "./retry";
import { createOpenAiProvider } from "./openai";
import { createDeepseekProvider } from "./deepseek";
import { createGeminiProvider } from "./gemini";

describe("Effect retry policy", () => {
  it("uses exponential delays and stops exactly at the retry budget", async () => {
    const events: RetryInfo[] = [];
    const failure = new OpenAI.APIError(503, undefined, "offline", undefined);
    let calls = 0;
    const task = retryOnTransientEffect(async () => { calls++; throw failure; }, {
      maxRetries: 2, baseDelayMs: 1000, jitter: false, onRetry: (info) => { events.push(info); },
    });
    await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* task.pipe(Effect.forkChild);
      yield* TestClock.adjust(0);
      expect(calls).toBe(1);
      yield* TestClock.adjust(999);
      expect(calls).toBe(1);
      yield* TestClock.adjust(1);
      expect(calls).toBe(2);
      yield* TestClock.adjust(2000);
      expect(yield* Fiber.join(fiber).pipe(Effect.flip)).toBe(failure);
      expect(calls).toBe(3);
      expect(events.map((e) => e.delayMs)).toEqual([1000, 2000]);
      expect(events.map((e) => e.attempt)).toEqual([1, 2]);
    }).pipe(Effect.provide(TestClock.layer())));
  });

  it("cancels the backoff without making another request", async () => {
    let calls = 0;
    const task = retryOnTransientEffect(async () => {
      calls++;
      throw new OpenAI.APIConnectionError({ message: "offline" });
    }, { baseDelayMs: 1000, jitter: false });
    await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* task.pipe(Effect.forkChild);
      yield* TestClock.adjust(0);
      expect(calls).toBe(1);
      yield* Fiber.interrupt(fiber);
      yield* TestClock.adjust("1 hour");
      expect(calls).toBe(1);
    }).pipe(Effect.provide(TestClock.layer())));
  });

  it("propagates interruption through the Promise provider adapter", async () => {
    const controller = new AbortController();
    let requestSignal: AbortSignal | undefined;
    const provider = withRetry({
      kind: "openai",
      complete: ({ signal }) => {
        requestSignal = signal;
        return new Promise(() => {});
      },
    });
    const completed = provider.complete({ model: "test", messages: [], reasoningEffort: "disabled", signal: controller.signal });
    const rejected = expect(completed).rejects.toThrow();
    controller.abort();
    await rejected;
    expect(requestSignal?.aborted).toBe(true);
  });

  it("applies the generation deadline to backoff as well as HTTP requests", async () => {
    vi.useFakeTimers();
    try {
      let calls = 0;
      const provider = withRetry({
        kind: "openai",
        complete: async () => {
          calls++;
          throw new OpenAI.APIError(503, undefined, "offline", undefined);
        },
      }, { baseDelayMs: 1000, jitter: false });
      const rejected = expect(runGeneration({
        provider,
        params: { model: "test", messages: [], reasoningEffort: "disabled" },
        scope: nullTracer.trace({ id: "test", name: "test" }),
        observation: { name: "generation" }, timeoutMs: 10,
      })).rejects.toThrow("Generation timed out after 10ms");
      await vi.advanceTimersByTimeAsync(10);
      await rejected;
      await vi.advanceTimersByTimeAsync(10_000);
      expect(calls).toBe(1);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  for (const [name, create] of Object.entries({ openai: createOpenAiProvider, gemini: createGeminiProvider, deepseek: createDeepseekProvider })) {
    it(`${name} disables SDK retries and passes the AbortSignal to the actual request`, async () => {
      let calls = 0;
      const client = new OpenAI({
        apiKey: "test-key", maxRetries: 8,
        fetch: async (_url, init) => {
          calls++;
          expect(init?.signal).toBeDefined();
          return new Response(JSON.stringify({ error: { message: "offline" } }), {
            status: 503, headers: { "content-type": "application/json" },
          });
        },
      });
      const provider = withRetry(create(client), { maxRetries: 1, baseDelayMs: 0 });
      await expect(provider.complete({ model: "test", messages: [], reasoningEffort: "disabled" })).rejects.toMatchObject({ status: 503 });
      expect(calls).toBe(2);
    });
  }
});
