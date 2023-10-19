"""大模型类"""


from base64 import encode
import json
import requests
from typing import Optional


class ChatModel: 
    """对话模型的基类"""

    def __init__(self): 
        raise NotImplementedError


class InternLM(ChatModel): 
    
    def __init__(
        self, 
        model_name: Optional[str] = "internlm-chat-20b", 
        host: str = None, 
        port: int = None, 
        max_length: Optional[int] = 4096, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7, 
        stream: Optional[bool] = False
    ): 
        self.model_name = model_name
        self.max_length: int = max_length
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.host = host
        self.port = port
        self.api: str = "http://{host}:{port}/v1/chat/completions".format(host=self.host, port=str(self.port))
        # self.prompt_template: str = PROMPT_TEMPLATE
        self.stream = stream
    
    def _call(self, prompt: str = ""): 
        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": prompt}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
            "stream": self.stream
        }
        return requests.post(self.api, json=payload)

    def chat(self, content: str = "") -> str: 
        resp = self._call(content)
        return resp.json().get("choices")[0].get("message").get("content")

    def stream_chat(self, content: str = "") -> str: 
        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": content}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
            "stream": self.stream
        }
        resp = requests.post(self.api, json=payload, stream=self.stream)
        # print(resp.iter_lines())
        result = []
        for chunk in resp.iter_lines(delimiter=b'data: '): 
            chunk = chunk.decode(encoding='utf-8').rstrip('\n\n')
            if chunk.strip() in ['', "[DONE]"]: 
                continue
            chunk = json.loads(chunk).get("choices")[0].get("delta").get("content")
            if chunk is None: 
                continue
            result.append(chunk)
            print(chunk, flush=True)
        return "".join(result)


if __name__ == '__main__': 
    llm = InternLM(
        host="172.21.4.23", 
        port=10375, 
        stream=True
    )

    # for i in range(1): 
    #     temp_prompt = """你好，请给我整理一份南京旅游攻略吧"""
    #     print("第{}轮生成".format(i), llm.chat(temp_prompt))

    temp_prompt = """你好！"""
    print(llm.stream_chat(temp_prompt))
    # print(llm.stream_chat(temp_prompt), flush=True)