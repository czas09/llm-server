"""对话模型接口：ChatGLM 风格"""


from fastapi import APIRouter
from loguru import logger


# 仿照 ChatGLM 风格接口：
# - /chat
# - /stream_chat
# - /batch_chat
chatglm_router = APIRouter()


@chatglm_router.post("/chat")
async def chat(): 
    raise NotImplementedError


@chatglm_router.post("/stream_chat")
async def stream_chat(): 
    raise NotImplementedError


@chatglm_router.post("/batch_chat")
async def batch_chat(): 
    raise NotImplementedError