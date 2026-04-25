// Test-harness JS standard library. Prepended to every scenario file before
// evaluation. Only add universally-useful helpers here — keep it small.

// Convenience: run multiple awaitables in parallel and resolve when all settle.
// Promise.all is available from the engine; this is just a named alias for
// readability when a scenario drives many robots at once.
globalThis.parallel = function parallel(promises) {
  return Promise.all(promises);
};

// Millisecond alias for scenario readability.
globalThis.ms = (n) => n;
globalThis.sec = (n) => n * 1000;
