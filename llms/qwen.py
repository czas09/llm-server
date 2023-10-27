# Qwen-7B-Chat
# Qwen-14B-Chat

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

from llms.base import BaseChatModel, BaseModelAdapter, BasePromptAdapter
from protocols import ChatMessage, Role
from config import MODEL_NAME, MODEL_PATH


class QwenModelAdapter(BaseModelAdapter): 
    """
    InternLM对话模型的模型适配
    """


    
    @property
    def model_type(self): 
        return "qwen"
    

class QwenPromptAdapter(BasePromptAdapter): 
    """
    Qwen对话模型的提示词适配

    参考链接：
    Qwen-7B-Chat: TODO
    Qwen-13B-Chat: TODO

    Qwen对话模型的提示词格式如下所示：
    <|im_start|>user\n{query0}<|im_end|>\n<|im_start|>assistant\n{response0}<|im_end|>\n<|im_start|>user\n{query0}<|im_end|>\n<|im_start|>assistant
    （遵循 OpenAI ChatML 格式）
    """

    def __init__(self): 
        self.system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        self.user_prompt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.assistant_prompt = "{}<|im_end|>\n"
        self.stop = {
            "strings": ["<|im_end|>"],
        }