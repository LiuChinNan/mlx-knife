# Project Context

## Purpose
This repository is a fork of https://github.com/mzau/mlx-knife. We continue the upstream focus on Apple Silicon local inference while improving codex-cli integration and reliability. The primary goals are to repair defects in the OpenAI-compatible endpoints, align behaviour with upstream Ollama when practical, and keep parity with the original CLI workflow so macOS developers obtain a stable local model manager.

## Tech Stack
- Python 3.9 as the baseline (compatible with Python 3.10–3.13)
- Apple Silicon macOS (M-series) with MLX and MLX-LM runtimes
- FastAPI with Uvicorn for the OpenAI-compatible HTTP server
- HuggingFace Hub for model discovery, downloads, and cache management
- Pytest, pytest-asyncio, pytest-timeout, pytest-mock, and psutil for testing
- Ruff, mypy (strict mode), and types-requests for tooling and static analysis

## Project Conventions

### Code Style
- Enforce Ruff with an 88-character limit plus bugbear and pyupgrade rules.
- Maintain Python 3.9 compatibility. Prefer `typing.Optional` style over `|` unions; avoid structural pattern matching.
- Follow semantic prefixes from `AGENTS.md` (for example `msTimeout`, `pctUsage`) to encode units or domains in names.
- Require explicit type annotations and pass mypy strict checks; disallow partially typed or untyped functions.
- Keep functions focused on a single intent with clear return values and explicit error handling.

### Architecture Patterns
- `mlx_knife.cli` owns argument parsing and command dispatch in a single-responsibility manner.
- `mlx_knife.mlx_runner` manages model loading, streaming, and resource cleanup through context managers.
- `cache_utils`, `hf_download`, and related modules isolate cache operations, HuggingFace interactions, and health checks.
- `mlx_knife.server` exposes the OpenAI-compatible API while delegating generation to the runner layer.
- Prefer composition and thin adapters; introduce helper modules when logic becomes complex rather than deep inheritance.

### Testing Strategy
- Run 166+ automated tests (unit and integration) with `pytest` and strict markers inside a 300-second timeout window.
- Use three-tier cache isolation: most tests rely on temporary directories so the user cache remains untouched.
- Mark long-running or model-dependent suites with `@pytest.mark.requires_model` or `@pytest.mark.server`; skip them by default in rapid feedback loops.
- Before opening a PR execute `ruff check --fix`, `mypy`, and `pytest`, and document the Python version plus hardware used.

### Git Workflow
- Treat `main` as the stable branch; contributors fork, create feature branches, and open pull requests.
- Commit messages must be CEFR B2 English subject lines with issue references (e.g., `(#1)`), followed by bullet-point bodies that summarise file-level changes.
- Large features or behavioural changes require an OpenSpec proposal first. Follow the “spec before implementation” rule and pass `openspec validate --strict`.

## Domain Context
- The tool markets itself as a local “knife” for managing MLX chat models: listing HuggingFace caches, downloading weights, running inference, and exposing chat workflows.
- Target users are individual researchers or developers who need rapid model swapping, health diagnostics, and a reliable CLI workflow.
- The OpenAI-compatible endpoints allow existing clients (such as Cursor or VS Code extensions) to reuse their tooling against local MLX models.
- Observability features like memory measurements, quantization metadata, and reasoning-aware streaming are essential for troubleshooting.

## Important Constraints
- Only Apple Silicon macOS is supported; MLX ≥0.29.0 and MLX-LM ≥0.27.0 are mandatory.
- The server assumes a single local user. There is no multi-tenant authorisation model.
- Runtime execution remains offline except when downloading from HuggingFace. Never upload user data elsewhere.
- Preserve Python 3.9 compatibility and the CEFR B2 naming standard for all identifiers, even while conversational replies stay in formal Taiwanese Mandarin.
- Every material change must respect the OpenSpec workflow with validated proposals and change tracking.
- For any review, planning, or architectural inquiry that is not explicitly scoped by the Producer, each defined team role must contribute an individual assessment and we must deliver a consolidated response.
- Enforce “one commit, one purpose”: scope each task to a single commit, and pair every task with its own planned or refined test coverage.
- Adopt a TDD mindset: define consensus test cases before implementation, add only new tests within a proposal, and never modify existing tests without a separate OpenSpec change.
- Whenever requirements are unclear, pause work and ask the Producer. Reasoning may be shared as options, but execution requires explicit confirmation unless told otherwise.
- Git commits must use CEFR B2 English subject lines, reference related issues in the title, and include bullet-point bodies that describe key improvements per file.
- Disallow `git commit -a`; stage files explicitly with `git add` to maintain precise control over each commit.
- Do not run `git commit` without Producer approval; request confirmation before composing each submission.

## External Dependencies
- HuggingFace Hub for cache management and model metadata.
- FastAPI and Uvicorn for `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.
- MLX and MLX-LM as the tensor runtime and tokenizer framework.
- Requests for HTTP operations outside the MLX stack.
- Psutil and related utilities for process and memory telemetry.
- Reference (read-only): `external/mlx` mirrors the upstream MLX framework documentation at https://ml-explore.github.io/mlx/.
- Reference (read-only): `external/ollama` mirrors the upstream Ollama project and its OpenAI-compatible endpoint behaviour.
