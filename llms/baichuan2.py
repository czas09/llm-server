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


class Baichuan2(BaseModel): 
    pass