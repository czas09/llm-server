# InternLM-Chat-7B
# InternLM-Chat-7B-v1.1
# InternLM-Chat-20B

from typing import Optional, List

from transformers import AutoModel
from peft import PeftModel

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from protocol import ChatMessage, Role
from config import MODEL_NAME, MODEL_PATH


class InternLMModelAdapter(BaseModelAdapter): 
    """
    InternLM对话模型的模型适配
    """
    
    @property
    def model_type(self): 
        return "internlm"


class InternLMPromptAdapter(BasePromptAdapter): 
    """
    InternLM对话模型的提示词适配

    参考链接：
    InternLM-Chat-7B: TODO
    InternLM-Chat-7B-v1.1: TODO
    InternLM-Chat-20B: TODO

    InternLM对话模型的提示词格式如下所示：
    <s><|User|>:{query0}<eoh>\n<|Bot|>:{response0}<eoa>\n<s><|User|>:{query1}<eoh>\n<|Bot|>:
    （其中 s = start，eoh = end-of-human，eoa = end-of-assistant）
    """

    def __init__(self): 
        self.system_prompt = ""    # InternLM对话模型没有支持系统提示词
        self.user_prompt = "<s><|User|>:{}<eoh>\n<|Bot|>:"
        self.assistant_prompt = "{}<eoa>\n"
        self.stop = {
            "strings": ["</s>", "<eoa>"],
        }
    
    def construct_prompt(self, messages: List[ChatMessage]) -> str: 
        prompt = self.system_prompt
        user_content = []
        i = 1
        for message in messages:
            role, content = message.role, message.content
            if role in [Role.USER, Role.SYSTEM]:
                user_content.append(content)
            elif role == Role.ASSISTANT:
                u_content = "\n".join(user_content)
                prompt += f"[Round {i}]\n\n{self.user_prompt.format(u_content)}"
                prompt += self.assistant_prompt.format(content)
                user_content = []
                i += 1
            else:
                raise ValueError(f"Unknown role: {role}")

        if user_content:
            u_content = "\n".join(user_content)
            prompt += f"[Round {i}]\n\n{self.user_prompt.format(u_content)}"

        return prompt