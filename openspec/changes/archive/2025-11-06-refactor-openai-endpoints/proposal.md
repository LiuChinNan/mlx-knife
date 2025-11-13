## Change Overview

Refactor the OpenAI-compatible FastAPI server so that streaming responses follow the official SSE contract, concurrent requests cannot invalidate active model runners, and error handling becomes deterministic. Current defects (documented in `1106_code_review.md`) break downstream clients and expose race conditions when switching models during active sessions.

### Why
- Streaming endpoints return `Content-Type: text/plain`, causing OpenAI SDKs and compliant HTTP clients to disconnect or misinterpret the stream.
- The global `_model_cache` flushes whenever a different model is requested, allowing one request to deallocate a runner that another request is using.
- End-of-stream payloads omit usage statistics; structured errors are inconsistent; missing tests allowed regressions to ship unnoticed.

### What Will Change
- Update SSE responses to emit `text/event-stream`, include usage summaries, and deliver a final `[DONE]` marker that mirrors the OpenAI contract.
- Introduce guarded model-runner lifecycle management (locking or reference tracking) so multiple requests can share or safely rotate models.
- Harden `/v1/models` and request validation to surface precise errors and align with Ollama-compatible metadata.
- Add structured logging hooks for streamed error chunks.
- Expand documentation to describe supported behaviours and limitations.

### Scope and Non-Goals
- Scope: API surface under `mlx_knife/server.py`, related runner helpers, and the minimal documentation/tests required to validate the fixes.
- Non-goals: Implement multi-tenant authentication, redesign CLI commands, or add non-MLX framework support.

### Risks and Mitigations
- Risk: New locking could introduce deadlocks. Mitigation: Use lightweight `asyncio.Lock` or reentrant guard with clear ownership.
- Risk: Streaming contract adjustments may break existing ad-hoc clients. Mitigation: Document changes and provide compatibility notes; adhere to the de-facto standard.

### Testing Strategy
- Add integration tests that spin up the FastAPI test client to verify SSE headers, chunk sequences, usage summaries, and `[DONE]` termination.
- Add concurrency-focused tests (mocking runner) to assert that switching models mid-stream raises a controlled error instead of tearing down the active runner.
- Ensure regression tests cover `/v1/models` filtering and detailed 404 / 400 responses.

### Timeline / Sequence
1. Stabilise model loading and caching semantics.
2. Update streaming generators and response schemas.
3. Add structured error/logging helpers.
4. Implement documentation updates and automated tests.
5. Run `openspec validate refactor-openai-endpoints --strict` before implementation handoff.
