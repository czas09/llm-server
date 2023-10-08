import sys
sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes import model_router, chat_router
from config import SERVICE_HOST, SERVICE_PORT, API_PREFIX


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix=API_PREFIX, tags=["Model"])
app.include_router(chat_router, prefix=API_PREFIX, tags=["Chat"])

# config:  
# if USE_SERVING_ENGINE == "vllm": 
#     app.include_router(chat_router)
# elif USE_SERVING_ENGINE == "lmdeploy": 
#     app.include_router(chat_router)


if __name__ == '__main__': 
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)