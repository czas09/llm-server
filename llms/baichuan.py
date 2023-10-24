# Baichuan-13B-Chat

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

from llms.base import BaseChatModel, BaseModelAdapter, BasePromptAdapter
from protocol import ChatMessage, Role
from config import config


class BaichuanModelAdapter(BaseModelAdapter): 
    """
    Baichuan对话模型的模型适配
    """

    def load_model_tokenizer(
        self, 
        model_path: str = config.MODEL_PATH, 
        adapter_path: Optional[str] = config.ADAPTER_PATH, 
        **kwargs): 

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

        if device == "cpu":
            model = model.float()

        # Tokenization后处理特殊字符
        tokenizer = self.post_tokenizer(tokenizer)

        if adapter_path is not None: 
            model = self.load_adapter_model(model, tokenizer, adapter_path, config_kwargs, **kwargs)
        
        quantize = kwargs.get("quantize", None)
        if quantize and quantize != 16: 
            logger.info(f"{quantize}位量化方式加载模型")
            model = model.quantize(quantize)
        
        if device == "cuda" and num_gpus == 1 and "device_map" not in config_kwargs:
            model.to(device)

        model.eval()

        return model, tokenizer
    
    def load_lora_model(self, model_path, adapter_path, model_kwargs): 
        return PeftModel.from_pretrained(model_path, adapter_path)
    
    @property
    def model_type(self): 
        return "baichuan"


class BaichuanPromptAdapter(BasePromptAdapter): 
    """
    Baichuan对话模型的提示词适配

    参考链接：
    Baichuan: TODO

    Baichuan对话模型的提示词格式如下所示：
    <reserved_102>{query0}<reserved_103>{response0}<reserved_102>{query1}<reserved_103>
    """

    def __init__(self): 
        self.system_prompt = ""    # Baichuan对话模型没有支持系统提示词
        self.user_prompt = "<reserved_102>{}<reserved_103>"
        self.assistant_prompt = "{}"
        self.stop = {
            "strings": ["<reserved_102>", "<reserved_103>"], 
            "token_ids": [195, 196], 
        }


class Baichuan(BaseChatModel): 

    def __init__(self): 
        raise NotImplementedError
    
    def get_model_adapter(): 
        """获取模型适配"""
        baichuan_model_adapter = BaichuanModelAdapter()
        return baichuan_model_adapter
    
    def get_prompt_adapter(): 
        """获取提示词适配"""
        baichuan_prompt_adapter = BaichuanPromptAdapter()
        return baichuan_prompt_adapter
    
    def load_model(): 
        raise NotImplementedError
    
