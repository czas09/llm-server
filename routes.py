import asyncio
import json
import secrets
from typing import Generator, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from models.chat_model import CHAT_MODEL
from config import (
    MODEL_NAME, 
    MODEL_PATH
)
from protocol import (
    ModelCard,
    ModelList,
    # ModelPermission,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    DeltaMessage,
    UsageInfo,
    Role,
)
from utils.utils import check_requests, create_error_response, ErrorCode


logger.add("./service.log", level='INFO')

model_router = APIRouter()
chat_router = APIRouter()                    # 仿照 ChatGLM 风格接口：/chat, /stream chat, /batch chat
openai_router = APIRouter(prefix="/chat")    # OpenAI API 风格接口：/v1/models, /v1/chat/completions


# ==============================================================================
# 仿照 ChatGLM 风格接口
# ==============================================================================

@model_router.post("/chat")
async def chat(): 
    raise NotImplementedError


@model_router.post("/stream_chat")
async def stream_chat(): 
    raise NotImplementedError


@model_router.post("/batch_chat")
async def batch_chat(): 
    raise NotImplementedError


# ==============================================================================
# OpenAI API 风格接口
# ==============================================================================

@model_router.get("/models")
async def show_available_models() -> ModelList: 
    logger.info("当前模型服务：")
    logger.info("    {}".format(MODEL_NAME))
    return ModelList(data=[ModelCard(id=MODEL_NAME, root=MODEL_NAME)])


@openai_router.post("/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse: 
    """Creates a completion for the chat message"""
    if len(request.messages) < 1 or request.messages[-1].role not in [Role.USER, Role.FUNCTION]:
        raise HTTPException(status_code=400, detail="Invalid request")

    # error_check_ret = check_requests(request)
    # if error_check_ret is not None:
    #     return error_check_ret

    messages = request.messages

    # TODO(@zyw): 为Qwen和InternLM模型实现Function call功能

    # 处理停止词：stop 和 stop_token_ids
    # stop settings
    stop, stop_token_ids = [], []
    if CHAT_MODEL.stop is not None: 
        stop_token_ids = CHAT_MODEL.stop.get("token_ids", [])
        stop = CHAT_MODEL.stop.get("strings", [])
    
    request.stop = request.stop or []
    if isinstance(request.stop, str): 
        request.stop = [request.stop]
    request.stop = list(set(stop + request.stop))

    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(stop_token_ids + request.stop_token_ids))

    gen_params = dict(
        model=request.model, 
        batch_prompts=batch_messages, 
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,    # 
        # echo=False,
        stream=request.stream,
        stop_token_ids=request.stop_token_ids,
        stop=request.stop,
    )

    logger.debug(f"==== request ====\n{gen_params}")

    # if request.stream:
    #     generator = chat_completion_stream_generator(request.model, gen_params, request.n)
    #     return StreamingResponse(generator, media_type="text/event-stream")
    
    batch_choices = []
    usage = UsageInfo()
    total_usage = 0
    for i in range(request.n): 
        batch_contents = CHAT_MODEL.generate_gate(gen_params)
        
        choices = []
        for content in batch_messages: 
            if content["error_code"]: 
                pass    # 处理错误

            finish_reason = "stop"
            message = ChatMessage(role=Role.ASSISTANT, content=content["text"])

            each_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in each_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
            # total_usage += each_usage

        choices.append(
            ChatCompletionResponseChoice(
                index=i, 
                batch_message=batch_messages, 
                finish_reason=finish_reason
            )
        )
        
        batch_choices.append(choices)

    return ChatCompletionResponse(model=request.model, batch_choices=batch_choices, usage=total_usage)