from typing import List

from llms import BaseChatModel, BaseModelAdapter, BasePromptAdapter
from protocols import ChatMessage, Role


class Llama2PromptAdapter(BasePromptAdapter): 
    """Llama-2 对话模型的提示词适配

    Llama-2 对话模型的提示词格式如下所示：
        <s>[INST] {query0} [/INST] {response0} </s><s>[INST] {query1} [/INST]
    """

    def __init__(self): 
        self.system_prompt = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""
        self.user_prompt = "[INST] {} "
        self.assistant_prompt = "[/INST] {} </s><s>"
        self.stop = {
            "strings": ["[INST]", "[/INST]"],
        }

    def construct_prompt(self, messages: List[ChatMessage]) -> str: 

        prompt = self.system_prompt
        for i, message in enumerate(messages):
            role, content = message.role, message.content
            if i == 0:
                prompt += content + " "
            else:
                if role == Role.USER:
                    prompt += self.user_prompt.format(content)
                else:
                    prompt += self.assistant_prompt.format(content)

        if messages[-1].role == Role.USER:
            prompt += "[/INST] "

        return prompt