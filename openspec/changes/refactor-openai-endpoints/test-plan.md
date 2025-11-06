# Test Planning Consensus (refactor-openai-endpoints)

## Existing Coverage Recap
- Streaming contract tests ensure SSE outputs include usage metadata and `[DONE]` markers.
- Runner cache tests verify reference counting and safe release behaviour.
- Model catalogue tests validate `/v1/models` metadata (type, context length).

## Planned Additions
### Task 5 — Structured Logging Hooks
- Add a unit-level test that loads and releases a stub runner, asserting `runner_loaded` and `runner_released` log events via `caplog`.
- Add a streaming error test that triggers a generator failure and verifies a `stream_error` log containing model identifier and mode.

## Execution Notes
- New tests live under `tests/unit/` or `tests/integration/` as appropriate; existing tests remain untouched.
- Logging assertions rely on structured `extra` fields (`record.event`, `record.fields`).
- If future tasks extend logging, append scenarios rather than modifying these baselines.
