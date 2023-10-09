# LLM-Server

## 功能点

* 支持模型 TODO
  * ChatGLM
  * Baichuan
  * Qwen
  * InternLM
  * Xverse
  * ...
  * 支持自定义模型 TODO
* 支持加载PEFT模型
  * LoRA
  * P-Tuning V2
* 接口参数格式
  * OpenAI API 格式：/v1/chat/completions
  * 仿 ChatGLM 格式：/chat、/stream_chat、/batch_chat
* 后端引擎
  * HF Transformers
  * vLLM
  * LMDeploy TODO

## 后端引擎

* HF Transformers
* vLLM
* LMDeploy TODO

## 代码结构

* 启动入口：app.py
* 配置参数：config.py

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

启动配置项设置

#### OpenAI API

调用方法

#### ChatGLM API

### 后端引擎为 vLLM

#### OpenAI API

#### ChatGLM API

## 待办事项

* 整合之前开发的 embedding-model-server
