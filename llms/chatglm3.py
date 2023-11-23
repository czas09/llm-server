# ChatGLM3-6B


import gc
import json
import os.path
import re
from typing import Optional, List
from click import command

import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.generation.logits_process import LogitsProcessor
from transformers.utils.versions import require_version
from peft import PeftModel
from loguru import logger

from llms.base import BaseChatModel, BaseModelAdapter, BasePromptAdapter
from llms.chatglm2 import ChatGLM2ModelAdapter, ChatGLM2
from protocols import ChatMessage, Role
from config import config


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLM3ModelAdapter(ChatGLM2ModelAdapter): 
    """ChatGLM3-6B 对话模型的模型适配模块"""

    @property
    def model_type(self): 
        return "chatglm3"


class ChatGLM3PromptAdapter(BasePromptAdapter): 
    """ChatGLM3-6B 对话模型的提示词适配模块
    
    ChatGLM3-6B 对话模型的提示词模板：
    TODO
    """

    def __init__(self): 
        # TODO(@zyw) 这里需要适配 ChatGLM3-6B 
        self.system_prompt = ""    # ChatGLM2对话模型没有支持系统提示词
        self.user_prompt = "问：{}\n\n答："
        self.assistant_prompt = "{}\n\n"
        self.stop = dict()
    

class ChatGLM3(ChatGLM2): 
    """ChatGLM3-6B 对话模型的交互接口"""

    def __init__(self): 
        self.model_adapter: ChatGLM3ModelAdapter = self._get_model_adapter()
        self.model, self.tokenizer = self._get_model_tokenizer()
        self.prompt_adapter: ChatGLM3PromptAdapter = self._get_prompt_adapter()
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        # self.prompt_name = 
        self.context_len: Optional[int] = config.CONTEXT_LEN
        self.stream_interval: Optional[int] = config.STREAM_INTERVERL
        self.use_streamer_v2: Optional[bool] = config.USE_STREAMER_V2
        self.do_construct_prompt: bool = False
        self.fix_tokenizer()

    def _get_model_tokenizer(self): 
        return self.model_adapter.load_model_tokenizer()
    
    def _get_model_adapter(self) -> ChatGLM3ModelAdapter: 
        """获取模型适配"""
        chatglm2_model_adapter = ChatGLM3ModelAdapter()
        return chatglm2_model_adapter

    def _get_prompt_adapter(self) -> ChatGLM3PromptAdapter: 
        """获取提示词适配"""
        internlm_prompt_adapter = ChatGLM3PromptAdapter()
        return internlm_prompt_adapter
    
    @torch.inference_mode()
    def _generate_stream(
            self, 
            model, 
            tokenizer, 
            params, 
            device: str, 
            context_len: int, 
            stream_interval: int = 2
    ):
        """ChatGLM3-6B 对话模型的文本生成接口实现"""

        prompt: List[ChatMessage] = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        # max_new_tokens = int(params.get("max_tokens", 256))
        max_tokens = int(params.get("max_tokens", 256))
        echo = params.get("echo", True)

        query, role = prompt[-1].content, prompt[-1].role
        history = [m.dict(exclude_none=True) for m in prompt[:-1]]

        inputs = tokenizer.build_chat_input(query, history=history, role=role)
        inputs = inputs.to(model.device)
        input_echo_len = len(inputs["input_ids"][0])

        eos_token_id = [
            tokenizer.eos_token_id, 
            tokenizer.get_command("<|user|>"), 
            # tokenizer.get_command("<|observation|>"), 
        ]

        gen_kwargs = dict(
            # max_length=max_new_tokens + input_echo_len,
            max_length=max_tokens, 
            do_sample=True if temperature > 1e-5 else False,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logits_processor=[InvalidScoreLogitsProcessor()],
        )
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature
        
        history.append(dict(
            role=role, 
            content=query
        ))

        total_len = 0
        # 这里使用的是 ChatGLM 模型官方实现的流式接口
        for total_ids in model.stream_generate(
                **inputs, 
                eos_token_id=eos_token_id, 
                **gen_kwargs
        ): 
            total_ids = total_ids.tolist()[0]
            total_len = len(total_ids)
            if echo:
                output_ids = total_ids[:-1]
            else:
                output_ids = total_ids[input_echo_len:-1]

            response = tokenizer.decode(output_ids)

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": None,
            }
        
        # TODO: ChatGLM stop when it reaches max length
        # Only last stream result contains finish_reason, we set finish_reason as stop
        # ChatGLM 流式生成只有在最后结束时 finish_reason 不为空
        ret = {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": "stop",
        }
        yield ret

        gc.collect()
        torch.cuda.empty_cache()