from typing import Optional

from peft import PeftModel

from models.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from config import MODEL_NAME

class BaichuanModelAdapter(BaseModelAdapter): 
    """
    Baichuan对话模型的LoRA适配
    """
    
    def load_lora_model(self, model_path, adapter_path, model_kwargs): 
        return PeftModel.from_pretrained(model_path, adapter_path))


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


class Baichuan(BaseModel): 

    def __init__(self): 
        raise NotImplementedError
    
    def get_model_adapter(): 
        """获取模型适配"""
        raise NotImplementedError
    
    def get_prompt_adapter(): 
        """获取提示词适配"""
        baichuan_prompt_adapter = BaichuanPromptAdapter()
        raise NotImplementedError
    
    def load_model(): 
        raise NotImplementedError


def get_baichuan_adapter(model_name: str): 


# def load_baichuan_model(
#         model_name: str, 
#         model_path: str, 
#         adapter_model_path: Optional[str] = None, 
#         quantize: Optional[int] = 16,    # 4, 8 or False
#         device: Optional[str] = 'device', 
# ): 
def load_baichuan_model(): 
    model_name = model_name.lower()

    baichuan_adapter = get_baichuan_adapter(MODEL_NAME)
    baichuan_model = baichuan_adapter.load_model(

    )
    raise NotImplementedError