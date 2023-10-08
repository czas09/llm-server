# ChatGLM2-6B

from typing import Optional

from transformers import AutoModel
from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
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