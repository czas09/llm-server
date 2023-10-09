import sys
sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes import model_router
from config import config


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO(@zyw): 整合之前开发好的文本embedding服务
# embedding-model-server...

if config.SERVING_ENGINE == "transformers": 
    from routes import openai_router, chat_router

    app.include_router(model_router, prefix="/v1", tags=["model"])
    app.include_router(openai_router, prefix="/v1", tags=["chat"])
    app.include_router(chat_router, tags=["chat"])

elif config.SERVING_ENGINE == "vllm": 
    from vllm_routes import openai_router, chat_router

    app.include_router(model_router, prefix="/v1", tags=["model"])
    app.include_router(openai_router, prefix="/v1", tags=["chat"])
    app.include_router(chat_router, tags=["chat"])

elif config.SERVING_ENGINE == "lmdeploy": 
    raise NotImplementedError("目前暂未支持lmdeploy")

else: 
    raise ValueError("SERVING_ENGINE must be one of [transformers, vllm, lmdeploy]")


if __name__ == '__main__': 
    import uvicorn

    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT)