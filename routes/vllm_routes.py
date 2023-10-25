"""对话模型接口：以 vLLM 为推理引擎"""


import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from models.chat_model import CHAT_MODEL
from config import config
from protocol import (
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


logger.add("./service.log", level='INFO')

# OpenAI API 风格接口：
# - /v1/chat/completions
openai_router = APIRouter(prefix="/chat")


@openai_router.post("/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse: 
    """参考 api-for-open-llm 与 vllm 两个项目中的相关实现"""

    logger.info(f"Received chat messages: {request.messages}")
    
    # TODO(@zyw): 完善入参校验
    # error_check_ret = await check_requests(request)
    # if error_check_ret is not None:
    #     return error_check_ret

    if len(request.messages) < 1 or request.messages[-1].role not in [Role.USER]:
        raise HTTPException(status_code=400, detail="Invalid request")

    # TODO(@zyw): 组装提示词的流程
    # 为百川和千问打补丁，尝试找到更好的写法
    if any(m in config.MODEL_NAME for m in ["baichuan", "qwen"]): 
        prompt = request.messages
    else: 
        prompt = CHAT_MODEL.prompt_adapter.construct_prompt(request.messages)

    request.max_tokens = request.max_tokens or 512
    token_ids, error_check_ret = await get_model_inputs(request, prompt, config.MODEL_NAME.lower())
    if error_check_ret is not None:
        return error_check_ret

    # 处理停止词：stop 和 stop_token_ids
    stop, stop_token_ids = [], []
    if CHAT_MODEL.prompt_adapter.stop is not None: 
        stop_token_ids = CHAT_MODEL.prompt_adapter.stop.get("token_ids", [])
        stop = CHAT_MODEL.prompt_adapter.stop.get("strings", [])
    
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
        # SamplingParams 是 vLLM 框架提供的采样参数列表
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

    # 这里的 CHAT_MODEL 是基于 vllm.AsyncLLMEngine 接口加载的
    result_generator = CHAT_MODEL.generate(
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

        # 返回的第一个 chunk 中只包含角色信息，content 字段为空
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
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:    # 假定有 n 条输入进行并行生成
                i = output.index
                output.text = output.text.replace("�", "")  # TODO: 修复Qwen模型问题
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                finish_reason = output.finish_reason
                if output.finish_reason is not None: 
                    delta_text = ""

                response_json = create_stream_response_json(
                    index=i, 
                    delta=DeltaMessage(content=delta_text, role=Role.ASSISTANT), 
                    finish_reason=finish_reason
                )
                yield f"data: {response_json}\n\n"

        # 为了兼容一部分 ChatGPT 客户端，最后生成一个 [DONE] 标记
        yield "data: [DONE]\n\n"

    # 流式生成
    if request.stream: 
        return StreamingResponse(completion_stream_generator(), media_type="text/event-stream")

    # 非流式生成
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await CHAT_MODEL.abort(request_id)
            return
        final_res = res

    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        output.text = output.text.replace("�", "")  # TODO: 修正千问模型的问题

        finish_reason = output.finish_reason
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
    max_input_tokens = CHAT_MODEL.max_model_len - request.max_tokens

    # TODO(@zyw): 各个对话模型在 PromprAdapter 类中自行实现 prompt_to_token_ids 接口 
    if isinstance(prompt, str):
        if getattr(request, "infilling", False):
            input_ids = CHAT_MODEL.engine.tokenizer(
                prompt,
                suffix_first=getattr(request, "suffix_first", False)
            ).input_ids
        else:
            input_ids = CHAT_MODEL.engine.tokenizer(prompt).input_ids[-max_input_tokens:]  # truncate left
    elif isinstance(prompt[0], int):
        input_ids = prompt[-max_input_tokens:]  # truncate left
    
    # TODO(@zyw): 这里也是对百川和千问打的补丁，尝试找到更好的写法
    else:
        if "baichuan" in model_name: 
            from llms.utils import build_baichuan_chat_input
            input_ids = build_baichuan_chat_input(
                CHAT_MODEL.engine.tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
            )
        elif "qwen" in model_name: 
            from llms.utils import build_qwen_chat_input
            input_ids = build_qwen_chat_input(
                CHAT_MODEL.engine.tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
            )
        else:
            raise ValueError(f"Model not supported yet: {model_name}")
    return input_ids, None
