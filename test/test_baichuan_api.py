
import requests
from typing import Optional



class ChatModel: 
    """对话模型的基类"""

    def __init__(self): 
        raise NotImplementedError


class Baichuan2(ChatModel): 
    
    def __init__(
        self, 
        model_name: Optional[str] = "baichuan-13b-chat", 
        host: str = None, 
        port: int = None, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7
    ): 
        self.model_name = model_name
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

    llm = Baichuan2(
        host="172.21.4.23", 
        port=10476, 
        max_length=4096, 
        temperature=0.01, 
        top_p=0.3
    )

    prompt = "你好，请帮我编写一份南京旅游攻略"
    print(llm.chat(prompt))