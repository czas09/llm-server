"""大模型类"""


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
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7, 
    ): 
        self.model_name = model_name
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.host = host
        self.port = port
        self.api: str = "http://{host}:{port}/v1/chat/completions".format(host=self.host, port=str(self.port))
        # self.prompt_template: str = PROMPT_TEMPLATE
    
    def _call(self, prompt: str = "", stream: bool = False): 
        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": prompt}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
            "stream": stream
        }
        return requests.post(self.api, json=payload, stream=stream)

    def chat(self, content: str = "") -> str: 
        resp = self._call(content)
        return resp.json().get("choices")[0].get("message").get("content")

    def stream_chat(self, content: str = "") -> str: 
        resp = self._call(content, stream=True)
        result = []
        for chunk in resp.iter_lines(delimiter=b'data: '): 
            chunk = chunk.decode(encoding='utf-8').rstrip('\n\n')
            if chunk.strip() in ['', "[DONE]"]: 
                continue
            chunk = json.loads(chunk).get("choices")[0].get("delta").get("content")
            if chunk is None: 
                continue
            result.append(chunk)
            yield "".join(result)


if __name__ == '__main__': 
    # 创建模型服务接口
    llm = InternLM(
        host="172.21.4.23", 
        port=10375, 
    )

    # 流式输出
    temp_prompt = """你好，请帮我编写一份南京旅游攻略"""
    pos = 0
    for result in llm.stream_chat(temp_prompt): 
        print(result[pos:], flush=True, end='')
        pos = len(result)
    print()