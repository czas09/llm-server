# LLM-Server

* [2023-11-09] 新增 seed 参数，用于固化模型推理随机性（分支：deploy-xj-20231109）
* [2023-11-07] 盐城部署（分支：xj-deploy-20231030）
* [2023-10-30] XJ大模型部署（分支：xj-deploy-20231030）

## 功能点

| 模型名称 | 接口格式 | 后端引擎 | 流式调用 | 加载 PEFT |
| ----------         | ----- | :---: | :---: | :---: |
| ChatGLM-6B         | ChatGLM style <br> OpenAI style (WIP) | HF Transformers | √ | √ |
| ChatGLM2-6b        | ChatGLM style <br> OpenAI style (WIP) | HF Transformers | √ | √ |
| Baichuan-13B-Chat  | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| Baichuan2-7B-Chat  | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| Baichuan2-13B-Chat | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| Qwen-7B-Chat       | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| Qwen-14B-Chat      | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| InternLM-Chat-7B   | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| InternLM-Chat-20B  | OpenAI style | HF Transformers <br> vLLM | √ | √ |
| XVERSE-13B-Chat    | WIP
| XVERSE-7B-Chat     | WIP
| AquilaChat2-7B     | WIP
| AquilaChat2-34B    | WIP
| Yi-6B              | WIP
| Yi-34B             | WIP

* 支持加载PEFT模型
  * LoRA
  * P-Tuning V2
* 接口参数格式
  * OpenAI API 格式：/v1/chat/completions
  * 仿 ChatGLM 格式：/chat、/stream_chat、/batch_chat

## 后端引擎

* HF Transformers
* vLLM
* LMDeploy TODO

## 代码结构

* 启动入口：app.py
* 配置参数：config.py + configs/*.ini

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

TODO 待整理
* 服务启动方法：后端为 transformers 或 vllm
* 服务调用方法：非流式、流式；cURL命令行调用 or openai python sdk or requests 调用
    * LangChain TODO

参考 configs/ 目录下的配置文件，填写 configs.ini；或者将 config.py 文件中 config.read() 中的参数改成对应文件路径

### 后端引擎为 HF Transformers

启动配置项设置，详见 configs/ 目录下的示例

#### OpenAI API

调用方法，详见 test/ 目录下的示例

#### ChatGLM API

### 后端引擎为 vLLM

#### OpenAI API

#### ChatGLM API

## 待办事项

* 优化服务启动方式，方便调整配置项的具体设置
* 设置各项参数的缺省值和校验逻辑
* 整合之前开发的 embedding-model-server
