import secrets
import time
from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field


class Role(str, Enum): 
    """对话角色"""
    SYSTEM = "system"          # 系统提示，用户提供，类似 instruction
    USER = "user"              # 用户输入问题
    ASSISTANT = "assistant"    # 对话模型回复
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
    model: str                                      # 模型名称
    messages: Optional[List[ChatMessage]]           # 输入文本（可包含对话历史）
    temperature: Optional[float] = 0.7              # 温度，用于控制生成文本的多样性或风格
    top_p: Optional[float] = 1.0                    # 用于控制模型生成下一个 token 时，在概率分布中的选择范围
    n: Optional[int] = 1                            # 返回候选文本数量
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None    # 停止词，文本生成过程在遇到这里指定的词汇后被截停
    stream: Optional[bool] = False                  # 是否流式传输
    presence_penalty: Optional[float] = 0.0         # 用于控制模型是否生成已生成的词汇：0 表示更有可能生成重复内容，1 表示生成完全不重复的内容
    frequency_penalty: Optional[float] = 0.0        # 用于控制模型是否生成常见词汇：0 表示更有可能生成常见词汇，1 表示完全避免生成常见词汇
    user: Optional[str] = None
    # functions: Optional[List[ChatFunction]] = None    # TODO(@zyw): 暂不支持 function call 相关功能
    # function_call: Union[str, Dict[str, str]] = "auto"

    # Additional parameters support for stop generation
    stop_token_ids: Optional[List[int]] = None
    repetition_penalty: Optional[float] = 1.1       # 作用与 stop 类似，但这里指定的是 token_ids

    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1                       # 作用与 top_p 类似，但限定的是 token 在概率分布中从大到小排名的前 k 个
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
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    # function_call: Optional[FunctionCallResponse] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage    # 在流式传输的情况下，生成内容放在 delta 字段中而不是 messages 字段
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