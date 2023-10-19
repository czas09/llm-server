"""大模型类"""


import requests
from typing import List, Optional

from pydantic import BaseModel


# TODO(@zyw): 合并两类模型服务接口的参数格式
class Params(BaseModel):
    """模型服务接口参数"""
    prompt: str = "hello"
    queries: List[str] = []
    history: List[List[str]] = []
    max_length: int = 8192
    top_p: float = 0.7
    temperature: float = 0.97
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample = True
    max_time: float = 60.0


class ChatModel: 
    """对话模型的基类"""

    def __init__(self): 
        raise NotImplementedError


class ChatGLM2(ChatModel): 

    def __init__(
        self, 
        model_name: Optional[str] = "chatglm2-6b", 
        host: str = None, 
        port: int = None, 
        max_length: Optional[int] = 8192, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7, 
        timeout: Optional[float] = 60.0
    ): 
        self.model_name = model_name
        self.max_length: int = max_length
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.host = host
        self.port = port
        # self.api: str = "http://{host}:{port}/chat".format(host=self.host, port=str(self.port))
        self.timeout = timeout
    
    def chat(self, content: str = "") -> str: 
        api: str = "http://{host}:{port}/chat".format(host=self.host, port=str(self.port))
        headers = {"Content-Type": "application/json", }
        params = dict(
            prompt=content, 
            history=[], 
            max_length=self.max_length, 
            top_p=self.top_p, 
            temperature=self.temperature, 
            repetition_penalty=1.0, 
            max_time=self.timeout
        )
        resp = requests.post(api, headers=headers, json=params)
        return resp.json().get("response")


class InternLM(ChatModel): 
    
    def __init__(
        self, 
        model_name: Optional[str] = "internlm-chat-20b", 
        host: str = None, 
        port: int = None, 
        max_length: Optional[int] = 4096, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7
    ): 
        self.model_name = model_name
        self.max_length: int = max_length
        self.temperature: float = temperature
        self.top_p: float = top_p
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
    # llm = InternLM(
    #     host="172.21.4.23", 
    #     port=10375
    # )

    # for i in range(1): 
    #     temp_prompt = """你好，请给我整理一份南京旅游攻略吧"""
    #     print("第{}轮生成".format(i), llm.chat(temp_prompt))

    llm = ChatGLM2(
        host="172.21.4.23", 
        port=10270, 
        timeout=2.0
    )

    prompt = "你好，请帮我编写一份南京旅游攻略"
    print(llm.chat(prompt))