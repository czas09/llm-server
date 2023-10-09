# LLM-Server

## 功能点

* 支持模型 TODO

  * ChatGLM
  * Baichuan
  * Qwen
  * InternLM
  * 自定义模型
* OpenAI API 风格接口参数
* ChatGLM 风格接口参数
* 模块化调用 TODO
* 后端引擎加速

## 后端引擎

* HF Transformers
* vLLM
* LMDeploy TODO

## 代码结构

启动入口：app.py
配置参数：config.py

```
llms/             不同大模型的对话提示词模板包装、模型加载与交互接口等实现
models/           不同类型模型（大模型、文本向量化模型）的加载逻辑
utils/            工具函数
app.py            主入口
config.py         配置项加载与预处理
configs.ini       配置项
protocol.py       参考 OpenAI API 的接口规范
routes.py         以transformers为后端的API路由实现
vllm_routes.py    以vLLM为后端的API路由实现
```

## 使用方法

### 后端引擎为 HF Transformers

#### OpenAI API

#### ChatGLM API

### 后端引擎为 vLLM

#### OpenAI API

#### ChatGLM API

## 待办事项

* 整合之前开发的 embedding-model-server
