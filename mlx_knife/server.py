# mlx_knife/server.py
"""
OpenAI-compatible API server for MLX models.
Provides REST endpoints for text generation with MLX backend.
"""

import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .cache_utils import (
    detect_framework,
    detect_model_type,
    get_model_path,
    is_model_healthy,
)
from .mlx_runner import MLXRunner

# Global model cache and configuration
_model_cache: Dict[str, MLXRunner] = {}
_runner_ref_counts: Dict[str, int] = {}
_cache_lock = threading.RLock()
_current_model_path: Optional[str] = None
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mlx-knife"
    permission: List = []
    context_length: Optional[int] = None



def _cleanup_stale_runners_locked(active_key: str) -> None:
    """Remove cached runners with no references while holding the lock."""
    stale_keys = [
        key for key, count in _runner_ref_counts.items()
        if count <= 0 and key != active_key
    ]
    for key in stale_keys:
        runner = _model_cache.pop(key, None)
        _runner_ref_counts.pop(key, None)
        if runner:
            try:
                runner.cleanup()
            except Exception:
                pass


def release_runner(model_key: str) -> None:
    """Release a cached runner reference and dispose if unused."""
    global _current_model_path
    if not model_key:
        return

    with _cache_lock:
        if model_key in _runner_ref_counts:
            _runner_ref_counts[model_key] -= 1
            if _runner_ref_counts[model_key] <= 0:
                runner = _model_cache.pop(model_key, None)
                _runner_ref_counts.pop(model_key, None)
                if runner:
                    try:
                        runner.cleanup()
                    except Exception:
                        pass
                if _current_model_path == model_key:
                    _current_model_path = None

        _cleanup_stale_runners_locked(active_key=model_key)


def get_or_load_model(model_spec: str, verbose: bool = False) -> Tuple[MLXRunner, str]:
    """Get model from cache or load it if not cached."""
    global _model_cache, _current_model_path

    # Use the existing model path resolution from cache_utils
    from .cache_utils import get_model_path

    try:
        model_path, model_name, commit_hash = get_model_path(model_spec)
        if model_path is None or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_spec} not found in cache")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model {model_spec} not found: {str(e)}")

    framework = detect_framework(model_path.parent.parent, model_name)
    if framework != "MLX":
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not a valid MLX model (Framework: {framework})")

    model_path_str = str(model_path)

    with _cache_lock:
        runner = _model_cache.get(model_path_str)

        if runner is None:
            if verbose:
                print(f"Loading model: {model_name}")

            runner = MLXRunner(model_path_str, verbose=verbose)
            runner.load_model()
            _model_cache[model_path_str] = runner

        _runner_ref_counts[model_path_str] = _runner_ref_counts.get(model_path_str, 0) + 1
        _current_model_path = model_path_str
        _cleanup_stale_runners_locked(active_key=model_path_str)

    return runner, model_path_str


async def generate_completion_stream(
    runner: MLXRunner,
    model_key: str,
    prompt: str,
    request: CompletionRequest
) -> AsyncGenerator[str, None]:
    """Generate streaming completion response."""
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())
    prompt_token_estimate = count_tokens(prompt)

    try:
        initial_response = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": None
                }
            ]
        }

        yield f"data: {json.dumps(initial_response)}\n\n"

        token_count = 0
        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=runner.get_effective_max_tokens(request.max_tokens or _default_max_tokens, interactive=False),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False  # Raw completion mode
        ):
            token_count += 1

            chunk_response = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": token,
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(chunk_response)}\n\n"

            # Check for stop sequences
            if request.stop:
                stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop]
                if any(stop in token for stop in stop_sequences):
                    break

        final_response = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "completion_tokens": token_count,
                "prompt_tokens": prompt_token_estimate,
                "total_tokens": prompt_token_estimate + token_count,
            }
        }

        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_response = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": "error"
                }
            ],
            "error": str(e)
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        release_runner(model_key)


async def generate_chat_stream(
    runner: MLXRunner,
    model_key: str,
    messages: List[ChatMessage],
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    prompt_contents = "\n\n".join(message.content for message in messages)
    prompt_token_estimate = count_tokens(prompt_contents)
    token_count = 0

    try:
        message_dicts = format_chat_messages_for_runner(messages)
        prompt = runner._format_conversation(message_dicts, use_chat_template=True)

        initial_response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }
            ]
        }

        yield f"data: {json.dumps(initial_response)}\n\n"

        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=runner.get_effective_max_tokens(request.max_tokens or _default_max_tokens, interactive=False),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False,  # Already applied in _format_conversation
            use_chat_stop_tokens=False  # Server mode shouldn't stop on chat markers
        ):
            token_count += 1
            chunk_response = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(chunk_response)}\n\n"

            if request.stop:
                stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop]
                if any(stop in token for stop in stop_sequences):
                    break

        final_response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "completion_tokens": token_count,
                "prompt_tokens": prompt_token_estimate,
                "total_tokens": prompt_token_estimate + token_count,
            }
        }

        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }
            ],
            "error": str(e)
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        release_runner(model_key)


def format_chat_messages_for_runner(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert chat messages to format expected by MLXRunner.
    
    Returns messages in dict format for the runner to apply chat templates.
    """
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text.split()) * 1.3)  # Approximation, convert to int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    print("MLX Knife Server starting up...")
    yield
    print("MLX Knife Server shutting down...")
    # Clean up model cache
    global _model_cache, _runner_ref_counts, _current_model_path
    with _cache_lock:
        try:
            for _runner in list(_model_cache.values()):
                try:
                    _runner.cleanup()
                except Exception:
                    pass
        finally:
            _model_cache.clear()
            _runner_ref_counts.clear()
            _current_model_path = None


# Create FastAPI app
from . import __version__

app = FastAPI(
    title="MLX Knife API",
    description="OpenAI-compatible API for MLX models",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint (OpenAI compatible)."""
    return {"status": "healthy", "service": "mlx-knife-server"}




@app.get("/v1/models")
async def list_models():
    """List available models (conservative, unchanged by Issue #31)."""
    from .cache_utils import MODEL_CACHE, cache_dir_to_hf

    model_list = []
    models = [d for d in MODEL_CACHE.iterdir() if d.name.startswith("models--")]

    for model_dir in models:
        model_name = cache_dir_to_hf(model_dir.name)
        framework = detect_framework(model_dir, model_name)

        if framework == "MLX" and is_model_healthy(model_name):
            # Only expose chat-capable models for the chat/completions API
            try:
                mtype = detect_model_type(model_dir, model_name)
            except Exception:
                mtype = "base"
            if mtype != "chat":
                continue
            # Get model context length (best effort)
            context_length = None
            try:
                model_path_tuple = get_model_path(model_name)
                if model_path_tuple and model_path_tuple[0]:
                    from .mlx_runner import get_model_context_length
                    context_length = get_model_context_length(str(model_path_tuple[0]))
            except Exception:
                pass

            model_list.append(ModelInfo(
                id=model_name,
                object="model",
                owned_by="mlx-knife",
                context_length=context_length
            ))

    return {"object": "list", "data": model_list}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    model_key = ""
    try:
        runner, model_key = get_or_load_model(request.model)

        # Handle array of prompts
        if isinstance(request.prompt, list):
            if len(request.prompt) > 1:
                raise HTTPException(status_code=400, detail="Multiple prompts not supported yet")
            prompt = request.prompt[0]
        else:
            prompt = request.prompt

        if request.stream:
            # Streaming response
            return StreamingResponse(
                generate_completion_stream(runner, model_key, prompt, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Non-streaming response
            completion_id = f"cmpl-{uuid.uuid4()}"
            created = int(time.time())

            generated_text = runner.generate_batch(
                prompt=prompt,
                max_tokens=runner.get_effective_max_tokens(request.max_tokens or _default_max_tokens, interactive=False),
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                use_chat_template=False
            )

            prompt_tokens = count_tokens(prompt)
            completion_tokens = count_tokens(generated_text)

            response = CompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "text": generated_text,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if model_key and not request.stream:
            release_runner(model_key)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    model_key = ""
    try:
        runner, model_key = get_or_load_model(request.model)

        if request.stream:
            # Streaming response
            return StreamingResponse(
                generate_chat_stream(runner, model_key, request.messages, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Non-streaming response
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())

            # Convert messages to dict format for runner
            message_dicts = format_chat_messages_for_runner(request.messages)
            
            # Let the runner format with chat templates
            prompt = runner._format_conversation(message_dicts, use_chat_template=True)

            generated_text = runner.generate_batch(
                prompt=prompt,
                max_tokens=runner.get_effective_max_tokens(request.max_tokens or _default_max_tokens, interactive=False),
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                use_chat_template=False  # Already applied in _format_conversation
            )

            # Token counting
            total_prompt = "\n\n".join([msg.content for msg in request.messages])
            prompt_tokens = count_tokens(total_prompt)
            completion_tokens = count_tokens(generated_text)

            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if model_key and not request.stream:
            release_runner(model_key)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    max_tokens: int = 2000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the MLX Knife server."""
    global _default_max_tokens
    _default_max_tokens = max_tokens

    print(f"Starting MLX Knife Server on http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print(f"Default max tokens: {'model-aware dynamic limits' if max_tokens is None else max_tokens}")

    uvicorn.run(
        "mlx_knife.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )
