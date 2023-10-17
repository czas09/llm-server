import configparser
from enum import auto
import os

from loguru import logger

from configs.model_paths import CHAT_MODEL_ROOTS, CHAT_MODEL_NAME_MAP, EMBEDDING_MODEL_NAME_MAP


logger.add("./service.log", level='INFO')

config = configparser.ConfigParser()
config.read("./configs/configs.ini", encoding='utf-8')


# TODO(@zyw): 各个参数项的缺省值
DEFAULT_CONFIGS = {
    # ==============================================================================
    # 服务配置选项
    # ==============================================================================
    "SERVICE_HOST": "0.0.0.0", 
    "SERVICE_PORT": 8000, 
    "API_PREFIX": "/1", 
    "API_KEYS": "xxx", 

    "MODEL_NAME": None, 
    "MODEL_PATH": None, 
    "ADAPTER_PATH": None, 
    "RESIZE_EMBEDDING": False, 

    "DEVICE": "cuda", 
    "DEVICE_MAP": None, 
    "GPUS": None, 
    "NUM_GPUS": 1, 

    "QUANTIZE": 16, 
    "LOAD_IN_8BIT": False, 
    "LOAD_IN_4BIT": False, 
    "USING_PTUNING_V2": False, 

    "CONTEXT_LEN": 2048, 
    "STREAM_INTERVERL": 2, 
    # PROMPT_NAME

    # ==============================================================================
    # 推理引擎相关
    # ==============================================================================
    # USE_VLLM
    "SERVING_ENGINE": "transformers", 

    # transformers 相关配置项
    "USE_STREAMER_V2": False, 

    # vLLM 相关配置项
    "TRUST_REMOTE_CODE": False, 
    "TOKENIZE_MODE": "auto", 
    "TENSOR_PARALLEL_SIZE": 1, 
    "DTYPE": "half", 
    "GPU_MEMORY_UTILIZATION": None, 
    "MAX_NUM_BATCHED_TOKENS": None, 
    "MAX_NUM_SEQS": 256, 
    "QUANTIZATION_METHOD": None, 
}


class Config: 
    """大模型服务启动项"""

    def __init__(self): 
        # ==============================================================================
        # 服务配置选项
        # ==============================================================================
        self.SERVICE_HOST = config.get("SERVICE", "host") \
            if config.get("SERVICE", "host") != "" else DEFAULT_CONFIGS["SERVICE_HOST"]
        self.SERVICE_PORT = config.getint("SERVICE", "port") \
            if config.get("SERVICE", "port") != "" else DEFAULT_CONFIGS["SERVICE_PORT"]
        self.API_PREFIX = config.get("SERVICE", "prefix") \
            if config.get("SERVICE", "prefix") != "" else DEFAULT_CONFIGS["API_PREFIX"]
        self.API_KEYS = config.get("SERVICE", "api_keys") \
            if config.get("SERVICE", "api_keys") != "" else DEFAULT_CONFIGS["API_KEYS"]

        # ==============================================================================
        # 模型配置选项
        # ==============================================================================
        self.MODEL_NAME = config.get("MODEL", "model_name").lower() \
            if config.get("MODEL", "model_name") != "" else DEFAULT_CONFIGS["MODEL_NAME"]
        chat_model_paths = [os.path.join(root, CHAT_MODEL_NAME_MAP[self.MODEL_NAME]) for root in CHAT_MODEL_ROOTS]
        chat_model_path = [path for path in chat_model_paths if os.path.isdir(path)][0]
        self.MODEL_PATH = config.get("MODEL", "model_path") \
            if config.get("MODEL", "model_path") != "" else chat_model_path
        self.ADAPTER_PATH = config.get("MODEL", "adapter_path") \
            if config.get("MODEL", "adapter_path") != "" else DEFAULT_CONFIGS["ADAPTER_PATH"]
        self.RESIZE_EMBEDDING = config.getboolean("MODEL", "resize_embedding") \
            if config.get("MODEL", "resize_embedding") != "" else DEFAULT_CONFIGS["RESIZE_EMBEDDING"]

        self.DEVICE = config.get("MODEL", "device") \
            if config.get("MODEL", "device") != "" else DEFAULT_CONFIGS["DEVICE"]
        self.DEVICE_MAP = config.get("MODEL", "device_map") \
            if config.get("MODEL", "device_map") != "" else DEFAULT_CONFIGS["DEVICE_MAP"]
        self.GPUS = config.get("MODEL", "gpus") \
            if config.get("MODEL", "gpus") != "" else DEFAULT_CONFIGS["GPUS"]
        self.NUM_GPUS = config.getint("MODEL", "num_gpus") \
            if config.get("MODEL", "num_gpus") != "" else DEFAULT_CONFIGS["NUM_GPUS"]

        self.QUANTIZE = config.getint("MODEL", "quantize") \
            if config.get("MODEL", "quantize") != "" else DEFAULT_CONFIGS["QUANTIZE"]
        self.LOAD_IN_8BIT = config.getboolean("MODEL", "load_in_8bit") \
            if config.get("MODEL", "load_in_8bit") != "" else DEFAULT_CONFIGS["LOAD_IN_8BIT"]
        self.LOAD_IN_4BIT = config.getboolean("MODEL", "load_in_4bit") \
            if config.get("MODEL", "load_in_4bit") != "" else DEFAULT_CONFIGS["LOAD_IN_4BIT"]
        self.USING_PTUNING_V2 = config.getboolean("MODEL", "using_ptuning_v2") \
            if config.get("MODEL", "using_ptuning_v2") != "" else DEFAULT_CONFIGS["USING_PTUNING_V2"]

        self.CONTEXT_LEN = config.getint("MODEL", "context_len") \
            if config.get("MODEL", "context_len") != "" else DEFAULT_CONFIGS["CONTEXT_LEN"]
        self.STREAM_INTERVERL = config.getint("MODEL", "stream_interval") \
            if config.get("MODEL", "stream_interval") != "" else DEFAULT_CONFIGS["STREAM_INTERVERL"]
        # self.PROMPT_NAME = config_parser.get("MODEL", "")    # TODO(@zyw)

        # ==============================================================================
        # 推理引擎相关
        # ==============================================================================
        # self.USE_VLLM = 
        self.SERVING_ENGINE = config.get("SERVING", "serving_engine") \
            if config.get("SERVING", "serving_engine") != "" else DEFAULT_CONFIGS["SERVING_ENGINE"]

        # transformers 相关配置项
        self.USE_STREAMER_V2 = config.getboolean("SERVING", "use_streamer_v2") \
            if config.get("SERVING", "use_streamer_v2") != "" else DEFAULT_CONFIGS["USE_STREAMER_V2"]

        # vLLM 相关配置项
        self.TRUST_REMOTE_CODE = config.getboolean("SERVING", "trust_remote_code") \
            if config.get("SERVING", "trust_remote_code") != "" else DEFAULT_CONFIGS["TRUST_REMOTE_CODE"]
        self.TOKENIZE_MODE = config.get("SERVING", "tokenize_mode") \
            if config.get("SERVING", "tokenize_mode") != "" else DEFAULT_CONFIGS["TOKENIZE_MODE"]
        self.TENSOR_PARALLEL_SIZE = config.getint("SERVING", "tensor_parallel_size") \
            if config.get("SERVING", "tensor_parallel_size") != "" else DEFAULT_CONFIGS["TENSOR_PARALLEL_SIZE"]
        self.DTYPE = config.get("SERVING", "dtype") \
            if config.get("SERVING", "dtype") != "" else DEFAULT_CONFIGS["DTYPE"]
        self.GPU_MEMORY_UTILIZATION = config.getfloat("SERVING", "gpu_memory_utilization") \
            if config.get("SERVING", "gpu_memory_utilization") != "" else DEFAULT_CONFIGS["GPU_MEMORY_UTILIZATION"]
        self.MAX_NUM_BATCHED_TOKENS = config.getint("SERVING", "max_num_batched_tokens") \
            if config.get("SERVING", "max_num_batched_tokens") != "" else DEFAULT_CONFIGS["MAX_NUM_BATCHED_TOKENS"]
        self.MAX_NUM_SEQS = config.getint("SERVING", "max_num_seqs") \
            if config.get("SERVING", "max_num_seqs") != "" else DEFAULT_CONFIGS["MAX_NUM_SEQS"]
        self.QUANTIZATION_METHOD = config.get("SERVING", "quantization_method") \
            if config.get("SERVING", "quantization_method") != "" else DEFAULT_CONFIGS["QUANTIZATION_METHOD"]

        # LMDeploy 相关配置项
        # TODO(@zyw)


config = Config()
logger.info("加载大模型服务配置项：{}".format(config.__dict__))
if config.GPUS: 
    if len(config.GPUS.split(",")) < config.NUM_GPUS: 
        raise ValueError("Larger --num_gpus ({}) than --gpus {}".format(config.NUM_GPUS, config.GPUS))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
