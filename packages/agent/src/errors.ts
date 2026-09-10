// Preserve SDK error identities so retry classification still sees HTTP status
// and connection errors. JavaScript callers can reject with any value.
export function toError(cause: unknown): Error {
  return cause instanceof Error ? cause : new Error(String(cause), { cause });
}

export function isAbortError(cause: unknown): boolean {
  return cause instanceof Error && cause.name === "AbortError";
}
