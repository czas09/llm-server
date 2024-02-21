"""对话模型接口：OpenAI 风格"""


import asyncio
import json
import secrets
from typing import Generator, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from models.chat_model import CHAT_MODEL
from config import config
from protocols import (
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
from utils import (
    create_error_response, 
    set_random_seed, 
    CHAT_MODEL_MAX_LEN_MAP
)


logger.add("./service.log", level='INFO')

# OpenAI API 风格接口：
# - /v1/chat/completions
openai_router = APIRouter(prefix="/chat")


@openai_router.post("/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse: 
    """Creates a completion for the chat message"""
    if len(request.messages) < 1 or request.messages[-1].role not in [Role.USER]:
        raise HTTPException(status_code=400, detail="Invalid request")

    # TODO(@zyw): 完善参数校验逻辑
    # error_check_ret = check_requests(request)
    # if error_check_ret is not None:
    #     return error_check_ret

    messages = request.messages

    # TODO(@zyw): 为Qwen和InternLM模型实现Function call功能

    # 处理停止词：stop 和 stop_token_ids
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
        prompt=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or CHAT_MODEL_MAX_LEN_MAP[config.MODEL_NAME],
        echo=False,
        stream=request.stream,
        stop_token_ids=request.stop_token_ids,
        stop=request.stop,
        repetition_penalty=request.repetition_penalty,
        # with_function_call=with_function_call,
    )

    logger.debug(f"==== request ====\n{gen_params}")

    # 为本次生成固定随机种子
    if request.seed: 
        set_random_seed(request.seed)

    if request.stream:
        generator = chat_completion_stream_generator(request.model, gen_params, request.n)
        return StreamingResponse(generator, media_type="text/event-stream")
    
    choices = []
    usage = UsageInfo()
    for i in range(request.n):
        content = CHAT_MODEL.chat(gen_params)
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])

        finish_reason = "stop"
        # if with_function_call:
        #     message, finish_reason = build_chat_message(content["text"], request.functions)
        # else:
        #     message = ChatMessage(role=Role.ASSISTANT, content=content["text"])
        message = ChatMessage(role=Role.ASSISTANT, content=content["text"])

        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=message,
                finish_reason=finish_reason,
            )
        )

        task_usage = UsageInfo.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, raw_request: Request
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    finish_stream_events = []
    for i in range(n): 
        # ======================================================================
        # First chunk with role (assistant)
        # ======================================================================
        choice_data = ChatCompletionResponseStreamChoice(
            index=i, 
            delta=DeltaMessage(role=Role.ASSISTANT),    # 返回的第一个 chunk 中只包含角色信息，content 字段为空
            finish_reason=None, 
        )
        chunk = ChatCompletionStreamResponse(
            id=_id, 
            model=model_name, 
            choices=[choice_data], 
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        for content in CHAT_MODEL.stream_chat(gen_params):
            # 连接中断
            if await raw_request.is_disconnected():
                asyncio.current_task().cancel()
                return

            # 发生错误
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # ==================================================================
            # 生成内容
            # ==================================================================
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None

            choice_data = ChatCompletionResponseStreamChoice(
                index=i, 
                delta=DeltaMessage(content=delta_text, role=Role.ASSISTANT), 
                finish_reason=content.get("finish_reason", "stop"), 
            )
            chunk = ChatCompletionStreamResponse(id=_id, choices=[choice_data], model=model_name)

            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue

            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content". 
    # 最后一个 delta（finish_reason 不为空）消息中不包含 content 字段
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    # 为了兼容一部分 ChatGPT 客户端，最后生成一个 [DONE] 标记
    yield "data: [DONE]\n\n"
