import sys
sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes import model_router
from config import SERVICE_HOST, SERVICE_PORT, API_PREFIX, SERVING_ENGINE


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

if SERVING_ENGINE == "transformers": 
    from routes import openai_router, chat_router

    app.include_router(model_router, prefix="/v1", tags=["model"])
    app.include_router(openai_router, prefix="/v1", tags=["chat"])
    app.include_router(chat_router, tags=["chat"])

elif SERVING_ENGINE == "vllm": 
    from vllm_routes import openai_router, chat_router

    app.include_router(model_router, prefix="/v1", tags=["model"])
    app.include_router(openai_router, prefix="/v1", tags=["chat"])
    app.include_router(chat_router, tags=["chat"])

elif SERVING_ENGINE == "lmdeploy": 
    raise NotImplementedError("lmdeploy integration is not implemented yet")

else: 
    raise ValueError("SERVING_ENGINE must be one of [transformers, vllm, lmdeploy]")


if __name__ == '__main__': 
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)