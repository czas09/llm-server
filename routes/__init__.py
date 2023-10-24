from config import config

from .model_routes import model_router
if "chatglm" in config.MODEL_NAME: 
    from .chatglm_routes import chatglm_router
from .chat_routes import openai_router
from .vllm_routes import openai_router
from .utils import load_model_on_gpus