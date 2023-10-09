import configparser
import os

from loguru import logger


logger.add("./service.log", level='INFO')

config_parser = configparser.ConfigParser()
config_parser.read("./configs.ini", encoding='utf-8')


# TODO(@zyw): 各个参数项的缺省值

class Config: 
    """大模型服务启动项"""

    def __init__(self): 
        # ==============================================================================
        # 服务配置选项
        # ==============================================================================
        self.SERVICE_HOST = config_parser.get("SERVICE", "host")
        self.SERVICE_PORT = config_parser.getint("SERVICE", "port")
        self.API_PREFIX = config_parser.get("SERVICE", "prefix")    # "/v1"
        self.API_KEYS = config_parser.get("SERVICE", "api_keys")
        # CHAT_ROUTE = configs.get("SERVICE", "chat_route")

        # ==============================================================================
        # 模型配置选项
        # ==============================================================================
        self.MODEL_NAME = config_parser.get("MODEL", "model_name").lower()
        self.MODEL_PATH = config_parser.get("MODEL", "model_path")    # TODO
        self.ADAPTER_PATH = config_parser.get("MODEL", "adapter_path")
        self.RESIZE_EMBEDDING = config_parser.get("MODEL", "resize_embedding")

        self.DEVICE = config_parser.get("MODEL", "device")
        self.DEVICE_MAP = config_parser.get("MODEL", "device_map")
        self.GPUS = config_parser.get("MODEL", "gpus")
        self.NUM_GPUS = config_parser.get("MODEL", "num_gpus")    # 形如：0,1,2

        self.QUANTIZE = config_parser.get("MODEL", "quantize")
        self.LOAD_IN_8BIT = config_parser.get("MODEL", "load_in_8bit")
        self.LOAD_IN_4BIT = config_parser.get("MODEL", "load_in_4bit")
        self.USING_PTUNING_V2 = config_parser.get("MODEL", "using_ptuning_v2")

        self.CONTEXT_LEN = config_parser.get("MODEL", "context_len")
        self.STREAM_INTERVERL = config_parser.get("MODEL", "stream_interval")
        # self.PROMPT_NAME = config_parser.get("MODEL", "")    # TODO(@zyw)

        # ==============================================================================
        # 推理引擎相关
        # ==============================================================================
        # self.USE_VLLM = 
        self.SERVING_ENGINE = config_parser.get("SERVING", "serving_engine")    # transformers (default), vllm or lmdeploy

        # transformers 相关配置项
        self.USE_STREAMER_V2 = config_parser.get("SERVING", "use_streamer_v2")

        # vLLM 相关配置项
        self.TRUST_REMOTE_CODE = config_parser.get("SERVING", "trust_remote_code")
        self.TOKENIZE_MODE = config_parser.get("SERVING", "tokenize_mode")
        self.TENSOR_PARALLEL_SIZE = config_parser.get("SERVING", "tensor_parallel_size")
        self.DTYPE = config_parser.get("SERVING", "dtype")
        self.GPU_MEMORY_UTILIZATION = config_parser.get("SERVING", "gpu_memory_utilization")
        self.MAX_NUM_BATCHED_TOKENS = config_parser.get("SERVING", "max_num_batched_tokens")
        self.MAX_NUM_SEQS = config_parser.get("SERVING", "max_num_seqs")
        self.QUANTIZATION_METHOD = config_parser.get("SERVING", "quantization_method")

        # LMDeploy 相关配置项
        # TODO(@zyw)


config = Config()
logger.info("加载大模型服务配置项：{}".format(config.__dict__))
if config.GPUS: 
    if len(config.GPUS.split(",")) < config.NUM_GPUS: 
        raise ValueError("Larger --num_gpus ({}) than --gpus {}".format(config.NUM_GPUS, config.GPUS))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS



# ==============================================================================
# 服务配置选项
# ==============================================================================
SERVICE_HOST = config_parser.get("SERVICE", "host")
SERVICE_PORT = config_parser.getint("SERVICE", "port")
API_PREFIX = config_parser.get("SERVICE", "prefix")    # "/v1"
# CHAT_ROUTE = configs.get("SERVICE", "chat_route")

# ==============================================================================
# 模型配置选项
# ==============================================================================
MODEL_NAME = config_parser.get("MODEL", "model_name").lower()
MODEL_PATH = config_parser.get("MODEL", "model_path")    # TODO
ADAPTER_PATH = config_parser.get("MODEL", "adapter_path")
RESIZE_EMBEDDING = None

QUANTIZE = config_parser.get("MODEL", "")    #

DEVICE = config_parser.get("MODEL", "")
DEVICE_MAP = config_parser.get("MODEL", "")
# GPUS = 
NUM_GPUS = config_parser.get("MODEL", "")

LOAD_IN_8BIT = config_parser.get("MODEL", "")
LOAD_IN_4BIT = config_parser.get("MODEL", "")
USING_PTUNING_V2 = config_parser.get("MODEL", "")
CONTEXT_LEN = config_parser.get("MODEL", "")
STREAM_INTERVERL = config_parser.get("MODEL", "")
PROMPT_NAME = config_parser.get("MODEL", "")
USE_STREAMER_V2 = config_parser.get("MODEL", "")

# ==============================================================================
# 推理引擎相关
# ==============================================================================
SERVING_ENGINE = config_parser.get("SERVING", "serving_engine")    # transformers, vllm, lmdeploy