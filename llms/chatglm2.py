# ChatGLM2-6B

import json
from typing import Optional, List

import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils.versions import require_version
from peft import PeftModel
from loguru import logger

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from protocol import ChatMessage, Role
from config import MODEL_NAME, MODEL_PATH


class ChatGLM2ModelAdapter(BaseModelAdapter): 
    """
    ChatGLM2对话模型的LoRA适配
    """
    
    @property
    def model_class(self): 
        return AutoModel

    @property
    def model_type(self): 
        return "chatglm2"


class ChatGLM2PromptAdapter(BasePromptAdapter): 
    """
    ChatGLM2对话模型的提示词适配

    参考链接：
    ChatGLM2-6B: TODO

    ChatGLM2对话模型的提示词格式如下所示：
        [Round 0]\n\n问：{query0}\n\n答：{response0}\n\n[Round 1]\n\n问：{query1}\n\n答：
    """

    def __init__(self): 
        self.system_prompt = ""    # ChatGLM2对话模型没有支持系统提示词
        self.user_prompt = "问：{}\n\n答："
        self.assistant_prompt = "{}\n\n"
        self.stop = dict()
    
    def construct_prompt(self, messages: List[ChatMessage]) -> str: 
        prompt = self.system_prompt
        user_content = []
        i = 1
        for message in messages:
            role, content = message.role, message.content
            if role in [Role.USER, Role.SYSTEM]:
                user_content.append(content)
            elif role == Role.ASSISTANT:
                u_content = "\n".join(user_content)
                prompt += f"[Round {i}]\n\n{self.user_prompt.format(u_content)}"
                prompt += self.assistant_prompt.format(content)
                user_content = []
                i += 1
            else:
                raise ValueError(f"Unknown role: {role}")

        if user_content:
            u_content = "\n".join(user_content)
            prompt += f"[Round {i}]\n\n{self.user_prompt.format(u_content)}"

        return prompt