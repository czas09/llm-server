from config import MODEL_NAME


def get_chat_model(): 
    """加载对话模型"""

    if "chatglm" in MODEL_NAME:    # TODO(zyw)
        from models.chatglm import load_chatglm_model
        model = load_chatglm_model()

    elif "baichuan" in MODEL_NAME:    # TODO(zyw)
        from models.baichuan import load_baichuan_model
        from models.baichuan import Baichuan
        model = Baichuan()

    elif "qwen" in MODEL_NAME:    # TODO(zyw)
        from models.qwen import load_qwen_model
        model = load_qwen_model()

    elif "internlm" in MODEL_NAME:    # TODO(zyw)
        from models.internlm import load_internlm_model
        model  = load_internlm_model()

    elif "xverse" in MODEL_NAME:    # TODO(zyw)
        from models.xverse import load_xverse_model
        model = load_xverse_model()
    
    return model


def get_embed_model():    # TODO(@zyw)
    """加载文本向量化模型"""
    raise NotImplementedError


def get_chat_model_with_vllm():    # TODO(@zyw)
    """以vLLM为后端引擎加载对话模型"""
    raise NotImplementedError


CHAT_MODEL = get_chat_model()
EXCLUDE_MODELS = ["baichuan-13b", "qwen"]  # model names for special processing
