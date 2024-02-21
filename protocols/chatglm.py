from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class Params(BaseModel):
    prompt: str = "hello"
    # queries: List[str] = []
    history: List[List[str]] = []
    max_length: int = 8192             # ChatGLM2 模型的默认上下文最大长度
    top_p: float = 0.7 
    temperature: float = 0.97
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    max_time: float = 60.0
    seed: Optional[int] = None


class Answer(BaseModel):
    status: int = 200
    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    response: str
    history: List[List[str]] = []


class BatchParams(BaseModel): 
    """针对批量调用接口的参数类型"""

    # prompt: str = "hello"
    queries: List[str] = []
    history: List[List[List[str]]] = []
    max_length: int = 8192             # ChatGLM2 模型的默认上下文最大长度
    top_p: float = 0.7 
    temperature: float = 0.97
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    max_time: float = 60.0


class BatchParamsForvLLM(BaseModel): 
    """针对批量调用接口的参数类型"""

    # TODO(@zyw): 入参对齐

    queries: List[str] = []
    history: List[List[List[str]]] = []
    max_length: int = 2048                          # 最大上下文长度
    top_p: float = 0.7 
    temperature: float = 0.97
    repetition_penalty: float = 1.0
    # num_beams: int = 1
    # do_sample: bool = True
    # max_time: float = 60.0    # vLLM 应该不支持，找其他解决方法

    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0         # 用于控制模型是否生成已生成的词汇：0 表示更有可能生成重复内容，1 表示生成完全不重复的内容
    frequency_penalty: Optional[float] = 0.0        # 用于控制模型是否生成常见词汇：0 表示更有可能生成常见词汇，1 表示完全避免生成常见词汇

    # Additional parameters supported by vLLM
    n: Optional[int] = 1                            # 返回候选文本数量
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    best_of: Optional[int] = None
    top_k: Optional[int] = -1                       # 作用与 top_p 类似，但限定的是 token 在概率分布中从大到小排名的前 k 个
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True


class BatchAnswer(BaseModel): 
    """针对批量调用响应的参数类型"""

    status: int = 200
    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    responses: List[str]
    # history: List[List[str]] = []