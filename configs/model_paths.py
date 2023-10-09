import os.path

model_path_schema = "{model_root}/{model_name}/{model_revision}"

# TODO(@zyw): 模型文件的存储路径
MODEL_ROOT = "/workspace/models"
MODEL_NAMES = [
    # 对话模型
    "chatglm-6b", 
    "chatglm2-6b", 
    "baichuan-13b-chat", 
    "baichuan2-7b-chat", 
    "baichuan2-13b-chat", 
    "internlm-chat-7b", 
    "internlm-chat-7b-v1-1", 
    "internlm-chat-20b", 
    "qwen-7b-chat", 
    "qwen-14b-chat", 
    # 文本向量化模型
    "text2vec-large-chinese", 
    "m3e-base", 
]
MODEL_PATH_MAP = {
    "baichuan-13b-chat": os.path.join(MODEL_ROOT, "baichuan-13b-chat")
}