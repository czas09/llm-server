# vLLM 魔改说明

* 支持 ChatGLM2-6B 模型

## 支持 ChatGLM2-6B 模型

首先确定当前虚拟环境 vllm 包的路径，比如： <br/>
`/IAOdata/environment/conda_envs/llm-server/lib/python3.8/site-packages/vllm/`

### 1、添加 ChatGLM2-6B 模型的 vLLM 实现

1. 把 `chatglm2` 目录下的 `chatglm.py` 文件复制到 `vllm/model_executor/models/` 目录下：

```bash
cd /path/to/vllm_utils
cp ./chatglm2/chatglm.py /path/to/env/vllm/model_executor/models/chatglm.py
```

2. 接着修改 `vllm/model_executor/models/` 目录的 `__init__.py` 文件：

添加模块导入语句：
```python
from vllm.model_executor.models.chatglm import ChatGLMModel
```

在 `__all__` 列表中添加模块暴露：
```python
__all__ = [
    ...
    "ChatGLMModel", 
]
```

3. 然后修改 vllm/model_executor/model_loader.py 文件

在模型注册字典 `_MODEL_REGISTRY` 中添加一行 `"ChatGLMModel": ChatGLMModel` 如下：
```python
_MODEL_REGISTRY = {
    ...
    "ChatGLMModel": ChatGLMModel, 
}
```

### 2、添加 ChatGLM2-6B 模型的配置文件

1. 把 `chatglm2` 目录下的 `chatglm_configs.py` 文件复制到 `vllm/transformers_utils/configs/` 目录下

```bash
cp ./chatglm2/chatglm_configs.py /path/to/env/vllm/transformers_utils/configs/chatglm_configs.py
```

2. 接着修改 `vllm/transformers_utils/configs/` 目录的 `__init__.py` 文件：

添加模块导入语句：
```python
from vllm.transformers_utils.configs.chatglm_configs import ChatGLMConfig
```

在 `__all__` 列表中添加模块暴露：
```python
__all__ = [
    ...
    "ChatGLMConfig", 
]
```

3. 然后修改 `vllm/transformers_utils/config.py` 文件：

在配置注册字典 `_CONFIG_REGISTRY` 中添加：
```python
_CONFIG_REGISTRY = {
    ...
    "chatglm": ChatGLMConfig, 
}
```

### 3、修改 model_loader

在 vllm/model_executor/model_loader.py 文件的 `_MODEL_REGISTRY` 字典中添加如下：

```python
_MODEL_REGISTRY = {
    ...
    "ChatGLMModel": ChatGLMModel, 
}
```

修改完成。