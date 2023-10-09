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
from utils import check_requests, create_error_response, ErrorCode


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
        prompt=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        stop_token_ids=request.stop_token_ids,
        stop=request.stop,
        repetition_penalty=request.repetition_penalty,
        # with_function_call=with_function_call,
    )

    logger.debug(f"==== request ====\n{gen_params}")

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
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role=Role.ASSISTANT),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=_id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        # with_function_call = gen_params.get("with_function_call", False)
        found_action_name = False
        for content in CHAT_MODEL.stream_chat(gen_params):
            if await raw_request.is_disconnected():
                asyncio.current_task().cancel()
                return

            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None

            messages = []
            # if with_function_call:
            #     if found_action_name:
            #         messages.append(build_delta_message(delta_text, "arguments"))
            #         finish_reason = "function_call"
            #     else:
            #         if previous_text.rfind("\nFinal Answer:") > 0:
            #             with_function_call = False

            #         if previous_text.rfind("\nAction Input:") == -1:
            #             continue
            #         else:
            #             messages.append(build_delta_message(previous_text))
            #             pos = previous_text.rfind("\nAction Input:") + len("\nAction Input:")
            #             messages.append(build_delta_message(previous_text[pos:], "arguments"))

            #             found_action_name = True
            #             finish_reason = "function_call"
            # else:
            #     messages = [DeltaMessage(content=delta_text, role=Role.ASSISTANT)]
            #     finish_reason = content.get("finish_reason", "stop")
            messages = [DeltaMessage(content=delta_text, role=Role.ASSISTANT)]
            finish_reason = content.get("finish_reason", "stop")

            chunks = []
            for m in messages:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=m,
                    finish_reason=finish_reason,
                )
                chunks.append(ChatCompletionStreamResponse(id=_id, choices=[choice_data], model=model_name))

            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.extend(chunks)
                continue

            for chunk in chunks:
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
