import asyncio

from llms import BaseChatModel
from config import MODEL_NAME, SERVING_ENGINE
from config import (
    MODEL_PATH, CONTEXT_LEN, QUANTIZATION_METHOD, 
    TOKENIZE_MODE, TRUST_REMOTE_CODE, DTYPE, TENSOR_PARALLEL_SIZE, 
    GPU_MEMORY_UTILIZATION, MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS
)
from utils import get_context_len


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
    """ get vllm generate engine for chat or completion. """
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer
    except ImportError:
        return None
    
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer_mode=TOKENIZE_MODE,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=DTYPE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=CONTEXT_LEN,
        quantization=QUANTIZATION_METHOD,
    )
    vllm_model = AsyncLLMEngine.from_engine_args(engine_args)

    # A separate tokenizer to map token IDs to strings.
    vllm_model.engine.tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=True,
    )

    if "internlm" in MODEL_NAME: 
        from llms.internlm import InternLM
        vllm_model.prompt_adapter = InternLM.prompt_adapter

    # engine_model_config = asyncio.run(vllm_model.get_model_config())
    vllm_model.engine.scheduler_config.max_model_len = CONTEXT_LEN
    vllm_model.max_model_len = CONTEXT_LEN

    return vllm_model


def get_chat_model_with_lmdploy():    # TODO(@zyw)
    """以LMDeploy为后端引擎加载对话模型"""
    raise NotImplementedError


if SERVING_ENGINE == "transformers": 
    CHAT_MODEL = get_chat_model()
elif SERVING_ENGINE == "vllm": 
    CHAT_MODEL_WITH_VLLM = get_chat_model_with_vllm()
elif SERVING_ENGINE == "lmdeploy": 
    CHAT_MODEL_WITH_LMDEPLOY = get_chat_model_with_lmdploy()
EXCLUDE_MODELS = ["baichuan-13b", "qwen"]  # model names for special processing
