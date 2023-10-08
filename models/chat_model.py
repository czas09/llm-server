from llms import BaseChatModel
from config import MODEL_NAME, SERVING_ENGINE


def get_chat_model(): 
    """加载对话模型"""

    if "chatglm2" in MODEL_NAME: 
        pass

    elif "chatglm" in MODEL_NAME:    # TODO(zyw)
        from llms.chatglm import load_chatglm_model
        model = load_chatglm_model()
    
    elif "baichuan2" in MODEL_NAME: 
        from llms import Baichuan2
        model = Baichuan2()

    elif "baichuan" in MODEL_NAME: 
        from llms import Baichuan
        model = Baichuan()

    elif "qwen" in MODEL_NAME:    # TODO(zyw)
        from llms.qwen import load_qwen_model
        model = load_qwen_model()

    elif "internlm" in MODEL_NAME: 
        from llms.internlm import InternLM
        model = InternLM()

    elif "xverse" in MODEL_NAME:    # TODO(zyw)
        from llms.xverse import load_xverse_model
        model = load_xverse_model()
    
    return model


def get_embed_model():    # TODO(@zyw)
    """加载文本向量化模型"""
    raise NotImplementedError


def get_chat_model_with_vllm():    # TODO(@zyw)
    """以vLLM为后端引擎加载对话模型"""
    raise NotImplementedError


def get_chat_model_with_lmdploy():    # TODO(@zyw)
    """以LMDeploy为后端引擎加载对话模型"""
    raise NotImplementedError


if SERVING_ENGINE == "transformers": 
    CHAT_MODEL = get_chat_model()
elif SERVING_ENGINE == "vllm": 
    CHAT_MODEL = get_chat_model_with_vllm()
elif SERVING_ENGINE == "lmdeploy": 
    CHAT_MODEL = get_chat_model_with_lmdploy()
EXCLUDE_MODELS = ["baichuan-13b", "qwen"]  # model names for special processing
