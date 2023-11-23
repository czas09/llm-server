from enum import IntEnum


CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 30
WORKER_API_TIMEOUT = 20


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


# TODO(@zyw): 模型最大容量
CHAT_MODEL_MAX_LEN_MAP = {
    "chatglm-6b": 2048, 
    "chatglm2-6b": 8192, 
    "chatglm3-6b": 8192, 
    "chatglm3-6b-32k": 32768, 
    "baichuan-13b-chat": 4096, 
    "baichuan2-7b-chat": 4096, 
    "baichuan2-13b-chat": 4096, 
    "qwen-7b-chat": 8192,       # 202308 旧版为 2048
    "qwen-14b-chat": 8192, 
    "internlm-chat-7b-v1-1": 2048, 
    "internlm-chat-20b": 4096, 
    "xverse-13b-chat": 2048, 
    "aquilachat2-7b": 2048, 
    "aquilachat2-34b": 2048, 
}