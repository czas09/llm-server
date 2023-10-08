import os.path

model_path_schema = "{model_root}/{model_name}/{model_revision}"

# 模型文件的存储路径
MODEL_ROOT = "/workspace/models"
MODEL_PATH_MAP = {
    "baichuan-13b-chat": os.path.join(MODEL_ROOT, "baichuan-13b-chat")
}