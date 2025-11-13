# api-server Specification

## Purpose
TBD - created by archiving change refactor-openai-endpoints. Update Purpose after archive.
## Requirements
### Requirement: OpenAI Streaming Contract
The API server MUST return Server-Sent Events that mirror the OpenAI streaming format.

#### Scenario: Streaming Completion Response
- **WHEN** a client requests `/v1/completions` with `stream=true`
- **THEN** the response MUST use `Content-Type: text/event-stream` and flush chunked `data:` frames with incremental text, usage statistics, and a terminal `data: [DONE]` record.

#### Scenario: Streaming Chat Response
- **WHEN** a client requests `/v1/chat/completions` with `stream=true`
- **THEN** each chunk MUST include `choices[0].delta` content, optional usage deltas, and the final event MUST set `finish_reason="stop"` before emitting `data: [DONE]`.

### Requirement: Concurrent Model Lifecycle Safety
The server MUST prevent one request from invalidating another active model runner.

#### Scenario: Parallel Model Requests
- **WHEN** Request A streams tokens from Model X and Request B selects Model Y
- **THEN** the loader MUST either serialize the swap until Request A completes or spawn a new runner without freeing Model X until all active requests finish.

### Requirement: Accurate Model Catalogue
The `/v1/models` endpoint MUST enumerate all healthy MLX models with clear metadata.

#### Scenario: Chat and Non-Chat Models
- **WHEN** the cache contains chat-capable and base MLX models
- **THEN** the response MUST include both with `context_length`, `owned_by`, and a `type` indicator (for example `chat`, `base`, `embedding`) so clients understand compatibility.

### Requirement: Structured Errors
The API MUST emit structured, actionable error payloads for both streaming and non-streaming failures.

#### Scenario: Missing Model Request
- **WHEN** a client requests a model that is absent or not MLX-compatible
- **THEN** the server MUST return `404` with a JSON body describing the issue, and streaming requests MUST send an error chunk followed by `data: [DONE]`.

