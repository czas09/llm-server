# Qwen-7B-Chat
# Qwen-14B-Chat

from typing import Optional

from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from config import MODEL_NAME, MODEL_PATH


class QwenModelAdapter(BaseModelAdapter): 
    """
    InternLM对话模型的模型适配
    """
    
    @property
    def model_type(self): 
        return "qwen"