# Xverse-13B-Chat

from typing import Optional

from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from config import MODEL_NAME, MODEL_PATH


class XverseModelAdapter(BaseModelAdapter): 
    """
    Xverse对话模型的模型适配
    """
    
    @property
    def model_type(self): 
        return "xverse"



class XversePromptAdapter(BasePromptAdapter): 
    """
    Xverse对话模型的提示词适配

    参考链接：
    Xverse-7B-Chat: TODO
    Xverse-13B-Chat: TODO

    Xverse对话模型的提示词格式如下所示：
    Human: {query0}\n\nAssistant: {response0}<|endoftext|>Human: {query1}\n\nAssistant:
    """

    def __init__(self): 
        self.system_prompt = ""    # Xverse对话模型没有支持系统提示词
        self.user_prompt = "Human: {}\n\nAssistant: "
        self.assistant_prompt = "{}<|endoftext|>"
        self.stop = dict()