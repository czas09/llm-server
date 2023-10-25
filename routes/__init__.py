from config import config

from .model_routes import model_router

if config.SERVING_ENGINE == "transformers": 
    from .chat_routes import openai_router

    # ChatGLM 与 ChatGLM2 模型额外支持 ChatGLM 格式接口
    if "chatglm" in config.MODEL_NAME: 
        from .chatglm_routes import chatglm_router

elif config.SERVING_ENGINE == "vllm": 
    from .vllm_routes import openai_router

else: 
    raise NotImplementedError

from .utils import load_model_on_gpus