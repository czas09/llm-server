# ChatGLM-6B

import json
from typing import Optional, List

import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils.versions import require_version
from peft import PeftModel
from loguru import logger

from llms.base import BaseModel, BaseModelAdapter, BasePromptAdapter
from protocol import ChatMessage, Role
from config import MODEL_NAME, MODEL_PATH


class ChatGLMModelAdapter(BaseModelAdapter): 
    """
    ChatGLM对话模型的模型适配
    """

    def load_model(self, 
                   model_path: str = MODEL_PATH, 
                   adapter_path: Optional[str] = None, 
                   **kwargs): 
        """使用Transformers作为后端引擎加载模型"""

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
                config_kwargs["device_map"] = "auto"    # TODO(@zyw): Baichuan多卡自动加载可能有bug，待测试
                # model_kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes

        # 量化相关配置项（使用bitsandbytes）
        if kwargs.get("load_in_8bit", False):
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")

            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("以8位量化类型加载模型")

        elif kwargs.get("load_in_4bit", False):
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")

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

        # Fix config (for Qwen)
        if hasattr(config, "fp16") and hasattr(config, "bf16"):
            setattr(config, "fp16", dtype == "half")
            setattr(config, "bf16", dtype == "bfloat16")
            config_kwargs.pop("torch_dtype", None)

        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        if use_ptuning_v2 and adapter_path: 
            with open(f'{adapter_path}/config.json', 'r') as f: 
                prefix_encoder_config = json.loads(f.read())

            config.pre_seq_len = prefix_encoder_config['pre_seq_len']
            config.prefix_projection = prefix_encoder_config['prefix_projection']

        # ======================================================================
        # 加载模型本体
        # ======================================================================
        # Load and prepare pretrained models (without valuehead).
        model = self.model_class.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            **config_kwargs
        )

        if device == "cpu":
            model = model.float()

        # post process for special tokens
        tokenizer = self.post_tokenizer(tokenizer)

        raise NotImplementedError
    
    def load_adapter_model(self, model, tokenizer, adapter_path, model_kwargs, **kwargs): 
        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        


    
    @property
    def model_class(self): 
        return AutoModel
    
    @property
    def model_type(self): 
        return "chatglm"
    

class ChatGLMPromptAdapter(BasePromptAdapter): 
    """
    ChatGLM对话模型的提示词适配

    参考链接：
    ChatGLM-6B: TODO

    ChatGLM对话模型的提示词格式如下所示：
        [Round 0]\n问：{query0}\n答：{response0}\n[Round 1]\n问：{query1}\n答：
    """

    def __init__(self): 
        self.system_prompt = ""    # ChatGLM对话模型没有支持系统提示词
        self.user_prompt = "问：{}\n答："
        self.assistant_prompt = "{}\n"
        self.stop = dict()
    
    def construct_prompt(self, messages: List[ChatMessage]) -> str: 
        prompt = self.system_prompt
        user_content = []
        i = 0
        for message in messages:
            role, content = message.role, message.content
            if role in [Role.USER, Role.SYSTEM]:
                user_content.append(content)
            elif role == Role.ASSISTANT:
                u_content = "\n".join(user_content)
                prompt += f"[Round {i}]\n{self.user_prompt.format(u_content)}"
                prompt += self.assistant_prompt.format(content)
                user_content = []
                i += 1
            else:
                raise ValueError(f"Unknown role: {role}")

        if user_content:
            u_content = "\n".join(user_content)
            prompt += f"[Round {i}]\n{self.user_prompt.format(u_content)}"

        return prompt