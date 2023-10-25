# 模型文件存储目录
CHAT_MODEL_ROOTS = [
    "/IAOdata/models",      # 在宿主机上的路径
    "/workspace/models",    # 在 Docker 镜像中的路径
]

# 对话模型文件版本
CHAT_MODEL_NAME_MAP = {
    "chatglm-6b": "chatglm-6b-20230515", 
    "chatglm2-6b": "chatglm2-6b-20230625", 
    "baichuan-13b-chat": "baichuan-13b-chat-20230718",    # 20230801
    "baichuan2-7b-chat": "baichuan2-7b-chat-20230906", 
    "baichuan2-13b-chat": "baichuan2-13b-chat-20230906", 
    "qwen-7b-chat": None, 
    "qwen-14b-chat": "qwen-14b-chat-20230924", 
    "internlm-chat-7b-v1-1": "internlm-chat-7b-v1-1-20230901",    # 20230822, 20230907
    "internlm-chat-20b": "internlm-chat-20b-20230920", 
    "xverse-13b-chat": "xverse-13b-chat-20230819", 
    "aquilachat2-7b": "aquilachat2-7b-20231011", 
    "aquilachat2-34b": "aquilachat2-34b-20231011", 
}

# TODO(@zyw): 模型最大容量
CHAT_MODEL_MAX_LEN_MAP = {
    "chatglm-6b": 2048, 
    "chatglm2-6b": 8192, 
    "baichuan-13b-chat": 4096, 
    "baichuan2-7b-chat": 4096, 
    "baichuan2-13b-chat": 4096, 
    "qwen-7b-chat": 2048, 
    "qwen-14b-chat": 2048, 
    "internlm-chat-7b-v1-1": 2048, 
    "internlm-chat-20b": 2048, 
    "xverse-13b-chat": 2048, 
    "aquilachat2-7b": 2048, 
    "aquilachat2-34b": 2048, 
}

# 文本向量化模型文件版本
EMBEDDING_MODEL_NAME_MAP = {
    "ernie-3.0-base-zh": "ernie-3.0-base-zh", 
    "ernie-3.0-nano-zh": "ernie-3.0-nano-zh", 
    "text2vec-large-chinese": "text2vec-large-chinese", 
    "m3e-base": "m3e-base-20230608", 
    "m3e-large": "m3e-large-20230621", 
}