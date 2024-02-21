# ChatGLM2-6B

import gc
import json
import os.path
import re
from typing import Optional, List

import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.generation.logits_process import LogitsProcessor
from transformers.utils.versions import require_version
from peft import PeftModel
from loguru import logger

from llms.base import BaseChatModel, BaseModelAdapter, BasePromptAdapter
from protocols import ChatMessage, Role
from config import config


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLM2ModelAdapter(BaseModelAdapter): 
    """
    ChatGLM2对话模型的LoRA适配
    """

    def load_model_tokenizer(
            self, 
            model_path: str = config.MODEL_PATH, 
            adapter_path: Optional[str] = config.ADAPTER_PATH, 
            **kwargs
    ):
        
        # ======================================================================
        # 加载tokenizer
        # ======================================================================
        tokenizer_kwargs = {
            "trust_remote_code": True, 
            "use_fast": False
        }
        tokenizer_kwargs.update(self.tokenizer_kwargs)
        if adapter_path is not None: 
            try: 
                tokenizer = self.tokenizer_class.from_pretrained(adapter_path, **tokenizer_kwargs)
            except OSError: 
                tokenizer = self.tokenizer_class.from_pretrained(model_path, **tokenizer_kwargs)
        else:
            tokenizer = self.tokenizer_class.from_pretrained(model_path, **tokenizer_kwargs)
        
        # ======================================================================
        # 处理模型加载相关配置项
        # ======================================================================
        config_kwargs = self.model_kwargs
        device = kwargs.get("device", "cuda")
        num_gpus = kwargs.get("num_gpus", 1)
        dtype = kwargs.get("dtype", "half")
        if device == "cuda": 
            if "torch_dtype" not in config_kwargs: 
                if dtype == "half":
                    config_kwargs["torch_dtype"] = torch.float16
                elif dtype == "bfloat16":
                    config_kwargs["torch_dtype"] = torch.bfloat16
                elif dtype == "float32":
                    config_kwargs["torch_dtype"] = torch.float32

            if num_gpus != 1: 
                config_kwargs["device_map"] = "auto"
                # model_kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes

        # 量化相关配置项（使用bitsandbytes）
        if kwargs.get("load_in_8bit", False):
            require_version("bitsandbytes>=0.37.0", "请更新bitsandbytes版本：pip install bitsandbytes>=0.37.0")

            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("以8位量化类型加载模型")

        elif kwargs.get("load_in_4bit", False):
            require_version("bitsandbytes>=0.39.0", "请更新bitsandbytes版本：pip install bitsandbytes>=0.39.0")
            require_version("peft>=0.4.0.dev0", "请更新peft版本：pip install git+https://github.com/huggingface/peft.git")

            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",    # NF4
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("以4位量化类型加载模型")

        if kwargs.get("device_map", None) == "auto":
            config_kwargs["device_map"] = "auto"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        if use_ptuning_v2 and adapter_path: 
            with open(f'{adapter_path}/config.json', 'r') as f: 
                prefix_encoder_config = json.loads(f.read())

            config.pre_seq_len = prefix_encoder_config['pre_seq_len']
            config.prefix_projection = prefix_encoder_config['prefix_projection']

        # ======================================================================
        # 加载模型
        # ======================================================================
        model = self.model_class.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            **config_kwargs
        )

        if device == "cpu":
            model = model.float()

        # Tokenization后处理特殊字符
        tokenizer = self.post_tokenizer(tokenizer)

        if adapter_path is not None: 
            model = self.load_adapter_model(model, tokenizer, adapter_path, config_kwargs, **kwargs)

        # ChatGLM2-6B 模型采用独立实现的量化方案
        quantize = kwargs.get("quantize", None)
        if quantize and quantize != 16:
            logger.info(f"将 ChatGLM2-6B 模型进行 {quantize} 位量化")
            model = model.quantize(quantize)

        if device == "cuda" and num_gpus == 1 and "device_map" not in config_kwargs:
            model.to(device)

        model.eval()

        return model, tokenizer

    def load_adapter_model(self, model, tokenizer, adapter_path, model_kwargs, **kwargs):
        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        resize_embeddings = kwargs.get("resize_embeddings", False)

        # 加载以 P-Tuning V2 方式微调的模型adapter
        if use_ptuning_v2:
            prefix_state_dict = torch.load(os.path.join(adapter_path, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            model.transformer.prefix_encoder.float()
        else:
            model = self.load_lora_model(model, adapter_path, model_kwargs)

    @property
    def model_class(self): 
        return AutoModel

    @property
    def model_type(self): 
        return "chatglm2"


class ChatGLM2PromptAdapter(BasePromptAdapter): 
    """
    ChatGLM2对话模型的提示词适配

    参考链接：
    ChatGLM2-6B: TODO

    ChatGLM2对话模型的提示词格式如下所示：
        [Round 0]\n\n问：{query0}\n\n答：{response0}\n\n[Round 1]\n\n问：{query1}\n\n答：
    """

    def __init__(self): 
        self.system_prompt = ""    # ChatGLM2对话模型没有支持系统提示词
        self.user_prompt = "问：{}\n\n答："
        self.assistant_prompt = "{}\n\n"
        self.stop = dict()
    
    def construct_prompt(self, messages: List[ChatMessage]) -> str: 
        """针对 OpenAI 风格接口"""

        prompt = self.system_prompt
        user_content = []
        i = 1
        for message in messages:
            role, content = message.role, message.content
            if role in [Role.USER, Role.SYSTEM]:
                user_content.append(content)
            elif role == Role.ASSISTANT:
                u_content = "\n".join(user_content)
                prompt += f"[Round {i}]\n\n{self.user_prompt.format(u_content)}"
                prompt += self.assistant_prompt.format(content)
                user_content = []
                i += 1
            else:
                raise ValueError(f"Unknown role: {role}")

        if user_content:
            u_content = "\n".join(user_content)
            prompt += f"[Round {i}]\n\n{self.user_prompt.format(u_content)}"

        return prompt
    
    def build_prompt(self, query: str, history: Optional[List[List[str]]] = None): 
        """针对 ChatGLM 风格接口
        
        TODO(@zyw): 这两个包装提示词的接口能否合并？
        """

        if history is None: 
            history = []

        prompt = ""
        # 在 prompt 中拼接历史消息
        for i, (old_query, response) in enumerate(history): 
            # 注意 ChatGLM 与 ChatGLM2 这两个模型的拼接模板上的
            prompt += "[Round {}]\n\n问：\n\n答：{}\n\n".format(i + 1, old_query, response)    # ChatGLM 模型的输入显式提供了问答轮数
        # 在 prompt 中拼接最新的 query 消息
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)


class ChatGLM2(BaseChatModel): 
    """ChatGLM2-6B 对话模型"""
    
    def __init__(self): 
        self.model_adapter: ChatGLM2ModelAdapter = self._get_model_adapter()
        self.model, self.tokenizer = self._get_model_tokenizer()
        self.prompt_adapter: ChatGLM2PromptAdapter = self._get_prompt_adapter()
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        # self.prompt_name = 
        self.context_len: Optional[int] = config.CONTEXT_LEN
        self.stream_interval: Optional[int] = config.STREAM_INTERVERL
        self.use_streamer_v2: Optional[bool] = config.USE_STREAMER_V2
        self.do_construct_prompt: bool = False
        self.fix_tokenizer()

    def _get_model_tokenizer(self): 
        return self.model_adapter.load_model_tokenizer()
    
    def _get_model_adapter(self) -> ChatGLM2ModelAdapter: 
        """获取模型适配"""
        chatglm2_model_adapter = ChatGLM2ModelAdapter()
        return chatglm2_model_adapter

    def _get_prompt_adapter(self) -> ChatGLM2PromptAdapter: 
        """获取提示词适配"""
        internlm_prompt_adapter = ChatGLM2PromptAdapter()
        return internlm_prompt_adapter

    def _process_response(response): 
        """ChatGLM 官方实现的后处理"""
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    @torch.inference_mode()
    def _generate_stream(
        self, 
        model,
        tokenizer,
        params,
        device,
        context_len=8192,
        stream_interval=2,    # TODO(@zyw): 入参对齐
    ): 
        """采用 ChatGLM 模型官方实现的流式接口"""

        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        echo = params.get("echo", True)

        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_echo_len = len(inputs["input_ids"][0])

        gen_kwargs = dict(
            max_length=max_new_tokens + input_echo_len,
            do_sample=True if temperature > 1e-5 else False,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logits_processor=[InvalidScoreLogitsProcessor()],
        )
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature

        total_len = 0
        # 这里使用的是 ChatGLM 模型官方实现的流式接口
        for total_ids in model.stream_generate(**inputs, **gen_kwargs): 
            total_ids = total_ids.tolist()[0]
            total_len = len(total_ids)
            if echo:
                output_ids = total_ids
            else:
                output_ids = total_ids[input_echo_len:]
            response = tokenizer.decode(output_ids)
            response = self._process_response(response)

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": None,
            }

        # TODO: ChatGLM stop when it reaches max length
        # Only last stream result contains finish_reason, we set finish_reason as stop
        # ChatGLM 流式生成只有在最后结束时 finish_reason 不为空
        ret = {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": "stop",
        }
        yield ret

        gc.collect()
        torch.cuda.empty_cache()
