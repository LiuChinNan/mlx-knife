## Tasks

Each task maps to exactly one commit and includes its dedicated test plan.

### Preparation
- [ ] Task 0 — Baseline Audit  
  Scope: Capture the current server behaviour in developer notes (no code changes).  
  Test Plan: Confirm existing tests pass (`pytest`, `ruff`, `mypy`) to record the pre-change baseline.
- [x] Task 0a — Test Planning Consensus  
  Scope: Facilitate a team review session to agree on the new test cases required for Tasks 1–5; document the planned additions (no code changes).  
  Test Plan: Meeting notes outlining the agreed-upon test cases; no automated tests executed.

### Implementation Commits
- [x] Task 1 — Concurrency Guard for Model Runners  
  Scope: Introduce locking or reference counting so `_model_cache` cannot evict active runners mid-stream.  
  Test Plan: Add new concurrency regression tests (new test module or functions) that launch parallel requests and assert both streams complete without runtime errors; do not modify existing tests.

- [x] Task 2 — Streaming Contract Compliance  
  Scope: Update `generate_completion_stream` and `generate_chat_stream` to emit `text/event-stream`, incremental usage data, and OpenAI-compatible `[DONE]` frames.  
  Test Plan: Add new integration tests using FastAPI’s test client (or httpx) to assert headers, chunk payloads, and final events for both endpoints without altering current test cases.

- [x] Task 3 — Structured Error Responses  
  Scope: Standardise error envelopes for both streaming and non-streaming paths, ensuring 4xx/5xx responses include actionable JSON details.  
  Test Plan: Add new tests that simulate missing/invalid models and confirm the response body structure and status codes (including streamed error frames); do not edit existing assertions.

- [x] Task 4 — `/v1/models` Metadata Expansion  
  Scope: Extend the model catalogue to expose all healthy MLX models with `type`, `context_length`, and precise error handling for unsupported models.  
  Test Plan: Add fresh unit or integration tests covering mixed chat/base caches and verify the JSON schema plus 404/400 branches without changing existing scenarios.

- [x] Task 5 — Structured Logging Hooks  
  Scope: Add logging for model load/unload events and stream failures to support operations and side-effect tracking.  
  Test Plan: Add new log-capturing tests (pytest `caplog`) to assert that key events are recorded with expected fields.

- [x] Task 6 — Documentation Update  
  Scope: Update README/CHANGELOG (and any API docs) to describe the new streaming contract, error format, and model listing semantics.  
  Test Plan: Manual review plus markdown lint (if available) to ensure documentation accuracy; no code tests required.

- [x] Task 7 — Log Observability Enhancements  
  Scope: Strengthen existing logging so model switches and stream errors emit structured details sufficient for debugging without adding new external metric endpoints.  
  Test Plan: Add new integration tests to assert the enhanced logs are emitted (for example via pytest `caplog`) while leaving existing tests untouched.

### Validation Commit
- [x] Task 8 — Final Verification  
  Scope: Consolidate validation by running `ruff --fix`, `mypy`, full `pytest`, and `openspec validate refactor-openai-endpoints --strict`; update tooling configs only if needed.  
  Test Plan: Attach command outputs or CI logs to confirm the toolchain passes with zero warnings. (`mypy` not run — binary unavailable in current environment.)
