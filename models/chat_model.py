from config import config


def get_chat_model(): 
    """加载对话模型"""

    if "chatglm2" in config.MODEL_NAME: 
        from llms import ChatGLM2
        model = ChatGLM2()

    elif "chatglm" in config.MODEL_NAME:    # TODO(zyw)
        from llms import ChatGLM
        model = ChatGLM()
    
    elif "baichuan2" in config.MODEL_NAME: 
        from llms import Baichuan2
        model = Baichuan2()

    elif "baichuan" in config.MODEL_NAME: 
        from llms import Baichuan
        model = Baichuan()

    elif "qwen" in config.MODEL_NAME:    # TODO(zyw)
        from llms import Qwen
        model = Qwen()

    elif "internlm" in config.MODEL_NAME: 
        from llms import InternLM
        model = InternLM()

    elif "xverse" in config.MODEL_NAME:    # TODO(zyw)
        from llms.xverse import load_xverse_model
        model = load_xverse_model()
    
    return model


def get_embed_model():    # TODO(@zyw)
    """加载文本向量化模型"""
    raise NotImplementedError


def get_chat_model_with_vllm():    # TODO(@zyw)
    """使用vLLM作为后端引擎加载对话模型"""
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer
    except ImportError:
        return None
    
    engine_args = AsyncEngineArgs(
        model=config.MODEL_PATH,
        tokenizer_mode=config.TOKENIZE_MODE,
        trust_remote_code=config.TRUST_REMOTE_CODE,
        dtype=config.DTYPE,
        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_num_batched_tokens=config.MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=config.MAX_NUM_SEQS,
        max_model_len=config.CONTEXT_LEN,
        quantization=config.QUANTIZATION_METHOD,
    )
    vllm_model = AsyncLLMEngine.from_engine_args(engine_args)

    # A separate tokenizer to map token IDs to strings.
    vllm_model.engine.tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=True,
    )

    if "internlm" in config.MODEL_NAME: 
        from llms.internlm import InternLMPromptAdapter
        vllm_model.prompt_adapter = InternLMPromptAdapter()

    # engine_model_config = asyncio.run(vllm_model.get_model_config())
    vllm_model.engine.scheduler_config.max_model_len = config.CONTEXT_LEN
    vllm_model.max_model_len = config.CONTEXT_LEN

    return vllm_model


def get_chat_model_with_lmdploy():    # TODO(@zyw)
    """以LMDeploy为后端引擎加载对话模型"""
    raise NotImplementedError


if config.SERVING_ENGINE == "transformers": 
    CHAT_MODEL = get_chat_model()
elif config.SERVING_ENGINE == "vllm": 
    CHAT_MODEL = get_chat_model_with_vllm()
elif config.SERVING_ENGINE == "lmdeploy": 
    CHAT_MODEL = get_chat_model_with_lmdploy()
# EXCLUDE_MODELS = ["baichuan-13b", "qwen"]  # model names for special processing
