import sys
sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes import model_router
from config import config, fake_argparser


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
    from routes import openai_router
    app.include_router(model_router, prefix="/v1", tags=["model list"])
    if "chatglm" in config.MODEL_NAME: 
        from routes import chatglm_router
        app.include_router(chatglm_router, tags=["chatglm"])
    app.include_router(openai_router, prefix="/v1", tags=["openai"])

elif config.SERVING_ENGINE == "vllm": 
    from routes import openai_router
    app.include_router(model_router, prefix="/v1", tags=["model list"])
    app.include_router(openai_router, prefix="/v1", tags=["openai"])

elif config.SERVING_ENGINE == "lmdeploy": 
    raise NotImplementedError("目前暂未支持lmdeploy！")

else: 
    raise ValueError("SERVING_ENGINE must be one of [transformers, vllm, lmdeploy]")


if __name__ == '__main__': 
    import uvicorn

    fake_argparser()

    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT)