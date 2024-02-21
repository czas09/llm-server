"""对话模型接口：ChatGLM 风格"""

# TODO(@zyw): 
# 完善模型服务启动配置的校验环节
#   1. 对接两类接口参数
#   2. vLLM 框架不支持 ChatGLM 系列模型


from datetime import datetime
import json
from typing import AsyncGenerator, List

import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from fastapi import Request, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from models.chat_model import CHAT_MODEL
from protocols import Params, BatchParams, Answer
# from routes.utils import load_model_on_gpus
from config import config


# 仿照 ChatGLM 风格接口：
# - /chat
# - /stream_chat
# - /batch_chat
chatglm_router = APIRouter()


def torch_gc(): 
    if torch.cuda.is_available(): 
        for i in range(config.NUM_GPUS): 
            with torch.cuda.device(i): 
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


# 非流式接口
@chatglm_router.post("/chat")
async def post_chat(params: Params) -> Answer: 
    answer = await create_chat(params)
    return answer


# 流式接口
@chatglm_router.post("/stream_chat")
async def post_stream_chat(params: Params) -> StreamingResponse: 
    return StreamingResponse(create_stream_chat(params))


# 批量调用接口（目前仅支持 ChatGLM2）
@chatglm_router.post("/batch_chat")
async def post_batch_chat(params: Params) -> List[str]: 
    answer_list = await batch_chat_chatglm2(params)
    return answer_list


async def create_chat(params: Params) -> Answer: 

    response, history = CHAT_MODEL.model.chat(    # 这里调用的是 ChatGLM 模型官方实现的对话接口
        CHAT_MODEL.tokenizer,
        params.prompt,
        history=params.history,
        max_length=params.max_length,
        top_p=params.top_p,
        temperature=params.temperature,
        repetition_penalty=params.repetition_penalty, 
        max_time=params.max_time
    )
    answer_ok = Answer(
        response=response, 
        history=history
    )
    torch_gc()
    return answer_ok


async def create_stream_chat(params: Params) -> AsyncGenerator: 

    for response, history in CHAT_MODEL.model.stream_chat(    # 这里调用的是 ChatGLM 模型官方实现的流式对话接口
        CHAT_MODEL.tokenizer,
        params.prompt,
        history=params.history,
        max_length=params.max_length,
        top_p=params.top_p,
        temperature=params.temperature,
        repetition_penalty=params.repetition_penalty, 
        max_time=params.max_time
    ):
        answer_ok = Answer(response=response, history=history)
        yield "\ndata: " + json.dumps(answer_ok.json())
    torch_gc()


async def batch_chat_chatglm2(params: BatchParams) -> List[str]: 
    """实现对话模型的批量调用能力
    TODO 对 ChatGLM 模型进行支持
    TODO 对 InternLM 模型进行支持
    """

    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    # 创建Logits处理器
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    # logits_processor = logits_processor
    
    gen_kwargs = dict(
        max_length=params.max_length,
        num_beams=params.num_beams,
        do_sample=params.do_sample,
        top_p=params.top_p,
        temperature=params.temperature,
        logits_processor=logits_processor,
    )

    # 对输入 query 进行组装各个模型的对话模板，形成可以激活对话能力的 prompt
    batch_inputs = []
    history = []
    for query in params.queries: 
        input = CHAT_MODEL.tokenizer.build_prompt(query, history=history)    # ChatGLM 1/2 是这里不同，build_prompt 是 2代实现的接口
        batch_inputs.append(input)

    # 将组装好的对话 prompt 编码成 token ids
    batch_input_ids = CHAT_MODEL.tokenizer(batch_inputs, return_tensors='pt', padding=True).to(torch.device('cuda'))
    # 这里采用的是 transformers 引擎加载模型，因此走 transformers 提供的 generate 接口
    batch_output_ids = CHAT_MODEL.model.generate(**batch_input_ids, **gen_kwargs).tolist()

    output_ids_list = []
    for input_ids, output_ids in zip(batch_input_ids["input_ids"], batch_output_ids):
        output_ids = output_ids[len(input_ids):]
        output_ids_list.append(torch.LongTensor(output_ids))

    batch_output_ids = torch.stack(output_ids_list)
    outputs = CHAT_MODEL.tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)

    responses = []
    for output in outputs:
        output = output.strip()
        responses.append(output)
    torch_gc()

    # TODO(@zyw): 使用 BatchAnswer 进行组装
    return responses
