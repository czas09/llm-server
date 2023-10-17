"""大模型类"""


import requests
from typing import Optional


class ChatModel: 
    """对话模型的基类"""

    def __init__(self): 
        raise NotImplementedError


class InternLM(ChatModel): 
    
    def __init__(
        self, 
        model_name: Optional[str] = "internlm-chat-7b-v1-1", 
        host: str = None, 
        port: int = None, 
    ): 
        self.model_name = model_name
        self.max_length: int = 4096
        self.temperature: float = 0.97
        self.top_p: float = 0.7
        self.host = host
        self.port = port
        self.api: str = "http://{host}:{port}/v1/chat/completions".format(host=self.host, port=str(self.port))
        # self.prompt_template: str = PROMPT_TEMPLATE
    
    def _call(self, prompt: str = ""): 
        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": prompt}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
        }
        return requests.post(self.api, json=payload)
    
    def chat(self, content: str = "") -> str: 
        resp = self._call(content)
        return resp.json().get("choices")[0].get("message").get("content")
    

if __name__ == '__main__': 
    llm = InternLM(
        host="172.21.4.23", 
        port=10470
    )

    for i in range(1): 
        temp_prompt = """你好，请给我整理一份南京旅游攻略吧"""
        print("第{}轮生成".format(i), llm.chat(temp_prompt))