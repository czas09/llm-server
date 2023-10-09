import secrets
import time
from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    # FUNCTION = "function"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


# class ModelPermission(BaseModel):
#     id: str = Field(default_factory=lambda: f"modelperm-{secrets.token_hex(12)}")
#     object: str = "model_permission"
#     created: int = Field(default_factory=lambda: int(time.time()))
#     allow_create_engine: bool = False
#     allow_sampling: bool = True
#     allow_logprobs: bool = True
#     allow_search_indices: bool = True
#     allow_view: bool = True
#     allow_fine_tuning: bool = False
#     organization: str = "*"
#     group: Optional[str] = None
#     is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "czas09"
    root: Optional[str] = None
    parent: Optional[str] = None
    # permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


# class ChatFunction(BaseModel):
#     name: str
#     description: Optional[str] = None
#     parameters: Optional[Any] = None


# class FunctionCallResponse(BaseModel):
#     name: Optional[str] = None
#     arguments: Optional[str] = None
#     thought: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str = None
    name: Optional[str] = None
    # functions: Optional[List[ChatFunction]] = None
    # function_call: Optional[FunctionCallResponse] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[List[ChatMessage]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # functions: Optional[List[ChatFunction]] = None
    # function_call: Union[str, Dict[str, str]] = "auto"

    # Additional parameters support for stop generation
    stop_token_ids: Optional[List[int]] = None
    repetition_penalty: Optional[float] = 1.1

    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    # finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{secrets.token_hex(12)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    batch_choices: List[List[ChatCompletionResponseChoice]]
    # choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    # function_call: Optional[FunctionCallResponse] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    # finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{secrets.token_hex(12)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo