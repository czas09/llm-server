# ChatGLM-6B

from typing import Optional

from transformers import AutoModel
from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from config import MODEL_NAME, MODEL_PATH


class ChatGLMModelAdapter(BaseModelAdapter): 
    """
    ChatGLM对话模型的模型适配
    """
    
    @property
    def model_class(self): 
        return AutoModel
    
    @property
    def model_type(self): 
        return "chatglm"
    


class ChatGLMPromptAdapter(BasePromptAdapter): 
    """
    ChatGLM对话模型的提示词适配

    参考链接：
    ChatGLM-6B: TODO

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