"""异步调用模型服务测试"""


import asyncio
import requests
import time
from typing import Optional

import aiohttp


class ChatModel: 
    """对话模型的基类"""

    def __init__(self): 
        raise NotImplementedError
    
    def _call(self, prompt: str = ""): 
        raise NotImplementedError


class InternLM(ChatModel): 
    """书生模型调用接口"""
    
    def __init__(
            self, 
            model_name: Optional[str] = "internlm-chat-20b", 
            host: str = None, 
            port: int = None, 
            max_length: Optional[int] = 2048, 
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
        """同步调用"""

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
        """同步调用"""

        resp = self._call(content)
        return resp.json().get("choices")[0].get("message").get("content")

    async def _async_call(self, prompt: str = ""): 
        """异步调用接口"""

        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": prompt}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
        }

        # 使用 aiohttp 库来进行异步 HTTP POST 请求
        async with aiohttp.ClientSession() as session: 
            async with session.post(self.api, json=payload) as response: 
                return await response.json()    # 等待响应返回
    

async def main(llm): 

    user_inputs = [
        "你好", 
        "请帮我解释一下巴以冲突的历史和成因", 
        "南京美食有哪些？", 
        "你叫什么名字？", 
    ]

    # 创建任务列表，其中的 llm._async_call 是异步方法
    tasks = [llm._async_call(input) for input in user_inputs]
    # 使用 asyncio.gather 异步执行上述任务
    results = await asyncio.gather(*tasks)
    # 获取结果，进行后续处理
    for result in results: 
        print("模型回复：", result)

    # 创建异步循环的另一种写法
    loop = asyncio.get_event_loop()     # 创建一个事件循环
    loop.run_until_complete(results)    # 执行上面的 asyncio.gather，等待全部完成
    loop.close()                        # 关闭事件循环


if __name__ == '__main__': 

    llm = InternLM(
        host="172.21.4.23", 
        port=10272, 
        temperature=0.01, 
        top_p=0.3
    )

    # ==========================================================================
    # 异步调用测试
    # ==========================================================================
    t0 = time.time()
    asyncio.run(main(llm))    # 执行异步调用的循环
    t1 = time.time()
    print(t1 - t0)

    # ==========================================================================
    # 同步调用测试
    # ==========================================================================
    user_inputs = [
        "你好", 
        "请帮我解释一下巴以冲突的历史和成因", 
        "南京美食有哪些？", 
        "你叫什么名字？", 
    ]

    t2 = time.time()
    for input in user_inputs: 
        print(llm.chat(input))
    t3 = time.time()
    print(t3 - t2)