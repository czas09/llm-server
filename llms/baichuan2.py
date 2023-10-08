from typing import Optional

from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from config import MODEL_NAME, MODEL_PATH


class Baichuan2ModelAdapter(BaseModelAdapter): 
    """
    Baichuan2对话模型的模型适配
    """
    
    def load_lora_model(self, model_path, adapter_path, model_kwargs): 
        return PeftModel.from_pretrained(model_path, adapter_path)

    @property
    def model_type(self): 
        return "baichuan2"
    

class Baichuan2PromptAdapter(BasePromptAdapter): 
    pass


class Baichuan2(BaseModel): 
    pass