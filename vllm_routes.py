import json
import base64
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from models.chat_model import CHAT_MODEL_WITH_VLLM
from config import (
    MODEL_NAME, 
    MODEL_PATH
)
# from api.apapter.react import (
#     check_function_call,
#     build_function_call_messages,
#     build_chat_message,
#     build_delta_message,
# )
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
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse: 
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    # error_check_ret = check_requests(request)
    # if error_check_ret is not None:
    #     return error_check_ret
    logger.info(f"Received chat messages: {request.messages}")

    if len(request.messages) < 1 or request.messages[-1].role not in [Role.USER, Role.FUNCTION]:
        raise HTTPException(status_code=400, detail="Invalid request")

    # with_function_call = check_function_call(request.messages, functions=request.functions)
    # if with_function_call and "qwen" not in config.MODEL_NAME.lower():
    #     raise HTTPException(status_code=400, detail="Invalid request format: functions only supported by Qwen-7B-Chat")

    # if with_function_call:
    #     if request.functions is None:
    #         for message in request.messages:
    #             if message.functions is not None:
    #                 request.functions = message.functions
    #                 break

    #     request.messages = build_function_call_messages(
    #         request.messages,
    #         request.functions,
    #         request.function_call,
    #     )

    # TODO
    # prompt = await get_gen_prompt(request, MODEL_NAME.lower())
    prompt = CHAT_MODEL_WITH_VLLM.prompt_adapter.construct_prompt(request.messages)
    request.max_tokens = request.max_tokens or 512
    token_ids, error_check_ret = await get_model_inputs(request, prompt, MODEL_NAME.lower())
    if error_check_ret is not None:
        return error_check_ret

    # 处理停止词：stop 和 stop_token_ids
    # stop settings
    stop, stop_token_ids = [], []
    if CHAT_MODEL_WITH_VLLM.prompt_adapter.stop is not None: 
        stop_token_ids = CHAT_MODEL_WITH_VLLM.prompt_adapter.stop.get("token_ids", [])
        stop = CHAT_MODEL_WITH_VLLM.stop.get("strings", [])
    
    request.stop = request.stop or []
    if isinstance(request.stop, str): 
        request.stop = [request.stop]
    request.stop = list(set(stop + request.stop))

    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(stop_token_ids + request.stop_token_ids))

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_generator = CHAT_MODEL_WITH_VLLM.generate(
        prompt if isinstance(prompt, str) else None,
        sampling_params,
        request_id,
        token_ids,
    )

    def create_stream_response_json(
        index: int,
        delta: DeltaMessage,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=delta,
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role=Role.ASSISTANT),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                choices=[choice_data],
                model=model_name
            )
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        found_action_name = False
        with_function_call = request.functions is not None
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")  # TODO: fix qwen decode
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                msgs = []
                # if with_function_call:
                #     if found_action_name:
                #         if previous_texts[i].rfind("\nObserv") > 0:
                #             break
                #         msgs.append(build_delta_message(delta_text, "arguments"))
                #         finish_reason = "function_call"
                #     else:
                #         if previous_texts[i].rfind("\nFinal Answer:") > 0:
                #             with_function_call = False

                #         if previous_texts[i].rfind("\nAction Input:") == -1:
                #             continue
                #         else:
                #             msgs.append(build_delta_message(previous_texts[i]))
                #             pos = previous_texts[i].rfind("\nAction Input:") + len("\nAction Input:")
                #             msgs.append(build_delta_message(previous_texts[i][pos:], "arguments"))

                #             found_action_name = True
                #             finish_reason = "function_call"
                # else:
                #     msgs = [DeltaMessage(content=delta_text, role=Role.ASSISTANT)]
                #     finish_reason = output.finish_reason
                msgs = [DeltaMessage(content=delta_text, role=Role.ASSISTANT)]
                finish_reason = output.finish_reason

                for m in msgs:
                    response_json = create_stream_response_json(index=i, delta=m, finish_reason=finish_reason)
                    yield f"data: {response_json}\n\n"

                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        delta=DeltaMessage(content="", role=Role.ASSISTANT),
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"

        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(), media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await CHAT_MODEL_WITH_VLLM.abort(request_id)
            return
        final_res = res


    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        output.text = output.text.replace("�", "")  # TODO: fix qwen decode

        finish_reason = output.finish_reason
        # if with_function_call:
        #     message, finish_reason = build_chat_message(output.text, request.functions)
        # else:
        #     message = ChatMessage(role=Role.ASSISTANT, content=output.text)
        message = ChatMessage(role=Role.ASSISTANT, content=output.text)

        choices.append(
            ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                finish_reason=finish_reason,
            )
        )

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming, but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(), media_type="text/event-stream")

    return response


async def get_model_inputs(request, prompt, model_name):
    max_input_tokens = CHAT_MODEL_WITH_VLLM.max_model_len - request.max_tokens
    if isinstance(prompt, str):
        if getattr(request, "infilling", False):
            input_ids = CHAT_MODEL_WITH_VLLM.engine.tokenizer(
                prompt,
                suffix_first=getattr(request, "suffix_first", False)
            ).input_ids
        else:
            input_ids = CHAT_MODEL_WITH_VLLM.engine.tokenizer(prompt).input_ids[-max_input_tokens:]  # truncate left
    elif isinstance(prompt[0], int):
        input_ids = prompt[-max_input_tokens:]  # truncate left
    # else:
    #     if "baichuan-13b" in model_name:
    #         input_ids = build_baichuan_chat_input(
    #             VLLM_ENGINE.engine.tokenizer,
    #             prompt,
    #             max_new_tokens=request.max_tokens,
    #         )
    #     elif "qwen" in model_name:
    #         input_ids = build_qwen_chat_input(
    #             VLLM_ENGINE.engine.tokenizer,
    #             prompt,
    #             max_new_tokens=request.max_tokens,
    #         )
    #     else:
    #         raise ValueError(f"Model not supported yet: {model_name}")
    return input_ids, None
