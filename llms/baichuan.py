# Baichuan-13B-Chat

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
from protocol import ChatMessage, Role
from config import MODEL_NAME, MODEL_PATH


class BaichuanModelAdapter(BaseModelAdapter): 
    """
    Baichuan对话模型的模型适配
    """
    
    def load_lora_model(self, model_path, adapter_path, model_kwargs): 
        return PeftModel.from_pretrained(model_path, adapter_path)
    
    @property
    def model_type(self): 
        return "baichuan"


class BaichuanPromptAdapter(BasePromptAdapter): 
    """
    Baichuan对话模型的提示词适配

    参考链接：
    Baichuan: TODO

    Baichuan对话模型的提示词格式如下所示：
    <reserved_102>{query0}<reserved_103>{response0}<reserved_102>{query1}<reserved_103>
    """

    def __init__(self): 
        self.system_prompt = ""    # Baichuan对话模型没有支持系统提示词
        self.user_prompt = "<reserved_102>{}<reserved_103>"
        self.assistant_prompt = "{}"
        self.stop = {
            "strings": ["<reserved_102>", "<reserved_103>"], 
            "token_ids": [195, 196], 
        }


class Baichuan(BaseChatModel): 

    def __init__(self): 
        raise NotImplementedError
    
    def get_model_adapter(): 
        """获取模型适配"""
        baichuan_model_adapter = BaichuanModelAdapter()
        return baichuan_model_adapter
    
    def get_prompt_adapter(): 
        """获取提示词适配"""
        baichuan_prompt_adapter = BaichuanPromptAdapter()
        return baichuan_prompt_adapter
    
    def load_model(): 
        raise NotImplementedError
    
