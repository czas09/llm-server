import configparser


configs = configparser.ConfigParser()
configs.read("./configs.ini", encoding='utf-8')

# ==============================================================================
# 服务配置选项
# ==============================================================================
SERVICE_HOST = configs.get("SERVICE", "host")
SERVICE_PORT = configs.getint("SERVICE", "port")
API_PREFIX = configs.get("SERVICE", "prefix")    # "/v1"
CHAT_ROUTE = configs.get("SERVICE", "chat_route")

# ==============================================================================
# 模型配置选项
# ==============================================================================
MODEL_NAME = configs.get("MODEL", "model_name")
MODEL_PATH = configs.get("MODEL", "model_path")
ADAPTER_MODEL_PATH = configs.get("MODEL", "")
QUANTIZE = configs.get("MODEL", "")
DEVICE = configs.get("MODEL", "")
DEVICE_MAP = configs.get("MODEL", "")
NUM_GPUS = configs.get("MODEL", "")
LOAD_IN_8BIT = configs.get("MODEL", "")
LOAD_IN_4BIT = configs.get("MODEL", "")
USING_PTUNING_V2 = configs.get("MODEL", "")
CONTEXT_LEN = configs.get("MODEL", "")
STREAM_INTERVERL = configs.get("MODEL", "")
PROMPT_NAME = configs.get("MODEL", "")