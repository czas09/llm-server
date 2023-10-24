# Baichuan2-7B-Chat
# Baichuan2-13B-Chat

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
from llms.baichuan import BaichuanModelAdapter
from protocol import ChatMessage, Role
from config import config


class Baichuan2ModelAdapter(BaichuanModelAdapter): 
    """
    Baichuan2对话模型的模型适配
    """

    @property
    def model_type(self): 
        return "baichuan2"
    

class BaichuanPromptAdapter(BasePromptAdapter): 
    """
    Baichuan2对话模型的提示词适配

    参考链接：
    Baichuan2-7B-Chat
    Baichuan2-13B-Chat: TODO

    Baichuan2对话模型的提示词格式如下所示：
    <reserved_102>{query0}<reserved_103>{response0}<reserved_102>{query1}<reserved_103>
    """

    def __init__(self): 
        self.system_prompt = ""    # Baichuan2对话模型没有支持系统提示词
        self.user_prompt = "<reserved_106>{}<reserved_107>"
        self.assistant_prompt = "{}"
        self.stop = {
            "strings": ["<reserved_106>", "<reserved_107>"],
            "token_ids": [195, 196],
        }


class Baichuan2(BaseChatModel): 
    pass