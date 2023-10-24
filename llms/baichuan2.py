# Baichuan2-7B-Chat
# Baichuan2-13B-Chat

from typing import Optional

from llms.base import BasePromptAdapter
from llms.baichuan import BaichuanModelAdapter, Baichuan
from config import config


class Baichuan2ModelAdapter(BaichuanModelAdapter): 
    """
    Baichuan2对话模型的模型适配
    """

    @property
    def model_type(self): 
        return "baichuan2"
    

class Baichuan2PromptAdapter(BasePromptAdapter): 
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
            "token_ids": [195, 196],    # Baichuan 和 Baichuan2 模型的 role token ids 是一样的
        }


class Baichuan2(Baichuan): 
    """Baichuan2对话模型"""

    def __init__(self): 
        self.model_adapter: Baichuan2ModelAdapter = self._get_model_adapter()
        self.model, self.tokenizer = self._get_model_tokenizer()
        self.prompt_adapter: Baichuan2PromptAdapter = self._get_prompt_adapter()
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        # self.prompt_name = 
        self.context_len: Optional[int] = config.CONTEXT_LEN
        self.stream_interval: Optional[int] = config.STREAM_INTERVERL
        self.use_streamer_v2: Optional[bool] = config.USE_STREAMER_V2
        self.fix_tokenizer()
    
    def _get_model_tokenizer(self): 
        return self.model_adapter.load_model_tokenizer()
    
    def get_model_adapter(): 
        """获取模型适配"""
        baichuan_model_adapter = Baichuan2ModelAdapter()
        return baichuan_model_adapter
    
    def get_prompt_adapter(): 
        """获取提示词适配"""
        baichuan_prompt_adapter = Baichuan2PromptAdapter()
        return baichuan_prompt_adapter
