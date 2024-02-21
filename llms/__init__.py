from typing import Dict

from .base import BaseChatModel, BaseModelAdapter, BasePromptAdapter
# from .aquila import Aquila
from .baichuan import Baichuan, BaichuanPromptAdapter
from .baichuan2 import Baichuan2, Baichuan2PromptAdapter
# from .chatglm import ChatGLM
from .chatglm2 import ChatGLM2
from .chatglm3 import ChatGLM3
from .internlm import InternLM, InternLMPromptAdapter
from .qwen import Qwen
# from .xverse import XVERSE


CHAT_MODEL_NAME_MAPPING: Dict[str, BaseChatModel] = {
    "base": BaseChatModel, 
    # "aquila": Aquila, 
    "baichuan": Baichuan, 
    "baichuan2": Baichuan2, 
    # "chatglm": ChatGLM, 
    "chatglm2": ChatGLM2, 
    "chatglm3": ChatGLM3, 
    "internlm": InternLM, 
    "qwen": Qwen, 
    # "xverse": XVERSE, 
}

PROMPT_ADAPTER_MAPPING: Dict[str, BasePromptAdapter] = {
    "baichuan": BaichuanPromptAdapter, 
    "baichuan2": Baichuan2PromptAdapter, 
    "internlm": InternLMPromptAdapter, 
}