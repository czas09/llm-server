from config import config

from .model_routes import model_router

if "chatglm" not in config.MODEL_NAME: 
    from .chat_routes import openai_router
    from .vllm_routes import openai_router
else:    # chatglm
    from .chatglm_routes import chatglm_router
    
from .utils import load_model_on_gpus