# InternLM-Chat-7B
# InternLM-Chat-7B-v1.1
# InternLM-Chat-20B

from typing import Optional

from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from config import MODEL_NAME, MODEL_PATH


class InternLMModelAdapter(BaseModelAdapter): 
    """
    InternLM对话模型的模型适配
    """
    
    @property
    def model_type(self): 
        return "internlm"