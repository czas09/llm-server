import json
import os
import sys
from typing import List, Optional, Union

import torch
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModel, 
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
)
from transformers.utils.versions import require_version

from protocol import ChatMessage, Role
from config import config
from utils import prepare_logits_processor, is_partial_stop, SERVER_ERROR_MSG
from utils.constants import ErrorCode


class BasePromptAdapter: 
    """对话模型提示词适配"""

    def __init__(self): 
        self.system_prompt: str = "You ara a helpful assistant!\n"
        self.user_prompt: str = "Human: {}\nAssistant: "
        self.assistant_prompt: str = "{}\n"
        self.stop = dict()
    
    def construct_prompt(self, messages: List[ChatMessage]) -> str: 
        """Covert messages into a prompt string.

        Args:
            messages (List[ChatMessage]): The conversation message in previous runs.

        Returns:
            string: formated prompt.
        """
        prompt = self.system_prompt
        user_content = []
        for message in messages: 
            role, content = message.role, message.content
            if role in [Role.USER, Role.SYSTEM]: 
                user_content.append(content)
            elif role == Role.ASSISTANT: 
                prompt += self.user_prompt.format("\n".join(user_content))
                prompt += self.assistant_prompt.format(content)
                user_content = []
            else: 
                raise ValueError(f"当前对话存在未知角色：{role}")
        
        if user_content: 
            prompt += self.user_prompt.format("\n".join(user_content))
        
        return prompt
    
    def get_model_inputs(self): 
        """将组装好的输入文本转换为对应的 token ids"""
        raise NotImplementedError


class BaseModelAdapter: 
    """模型适配（LoRA）"""
    
    def load_model_tokenizer(self, model_path: str, adapter_path: Optional[str] = None, **kwargs): 
        """使用Transformers作为后端引擎加载本地模型文件
        TODO(@zyw): 每个模型各自实现
        
        加载 tokenizer
        数据类型
        量化 load_in_8bit load_in_4bit
        模型配置 autoconfig
        lora和adapter
        加载模型
        执行量化

        """
        model_path = self.default_model_path if model_path is None else model_path
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": False}
        tokenizer_kwargs.update(self.tokenizer_kwargs)

        if adapter_path is not None: 
            try: 
                tokenizer = self.tokenizer_class.from_pretrained(adapter_path, **tokenizer_kwargs)
            except: 
                tokenizer = self.tokenizer_class.from_pretrained(model_path, **tokenizer_kwargs)
        else: 
            tokenizer = self.tokenizer_class.from_pretrained(model_path, **tokenizer_kwargs)
        
        config_kwargs = self.model_kwargs
        device = kwargs.get("device", "cuda")
        num_gpus = kwargs.get("num_gpus", 1)
        if device == "cuda": 
            if "torch_dtype" not in config_kwargs: 
                config_kwargs["torch_dtype"] = torch.float16
            if num_gpus != 1: 
                config_kwargs["device_map"] = "auto"
                # model_kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
        
        # Quantization configuration (using bitsandbytes library)
        if kwargs.get("load_in_8bit", False): 
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")

            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, 
                llm_int8_threshold=0.0, 
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("Quantizing model to 8 bit.")
        
        elif kwargs.get("load_in_4bit", False): 
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")

            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("Quantizing model to 4 bit.")

        if kwargs.get("device_map", None) == "auto":
            config_kwargs["device_map"] = "auto"
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        if use_ptuning_v2 and adapter_path:
            prefix_encoder_file = open(f'{adapter_path}/config.json', 'r')
            prefix_encoder_config = json.loads(prefix_encoder_file.read())
            prefix_encoder_file.close()

            config.pre_seq_len = prefix_encoder_config['pre_seq_len']
            config.prefix_projection = prefix_encoder_config['prefix_projection']

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
        is_chatglm = "chatglm" in str(type(model))

        if adapter_path is not None:
            model = self.load_adapter_model(model, tokenizer, adapter_path, is_chatglm, config_kwargs, **kwargs)

        if is_chatglm or "baichuan" in str(type(model)) or "xverse" in str(type(model)):
            quantize = kwargs.get("quantize", None)
            if quantize and quantize != 16:
                logger.info(f"Quantizing model to {quantize} bit.")
                model = model.quantize(quantize)

        if device == "cuda" and num_gpus == 1 and "device_map" not in config_kwargs:
            model.to(device)

        # inference mode
        model.eval()

        return model, tokenizer
    
    def load_lora_model(self, model, adapter_path, model_kwargs): 
        model = PeftModel.from_pretrained(
            model, 
            adapter_path, 
            torch_dtype=model_kwargs.get("torch_dtype", torch.float16), 
        )
        return model

    def post_tokenizer(self, tokenizer):
        # TODO(@zyw): Tokenization后处理
        return tokenizer
    
    @property
    def model_type(self): 
        raise NotImplementedError

    @property
    def model_name(self): 
        return config.MODEL_NAME
    
    @property
    def model_path(self): 
        return config.MODEL_PATH

    @property
    def model_class(self):
        return AutoModelForCausalLM

    @property
    def model_kwargs(self):
        return {}

    @property
    def tokenizer_class(self):
        return AutoTokenizer

    @property
    def tokenizer_kwargs(self):
        return {}


class BaseChatModel: 

    def __init__(self): 
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None
        self.context_len = Optional[int] = None
        self.stream_interval: Optional[int] = 2
        # self.prompt_name: Optional[str] = None
        # self.use_streamer_v2: Optional[bool] = False    # TODO(@zyw): Transformers提供的流式实现
        self.model_adapter: BaseModelAdapter = None
        self.prompt_adapter: BasePromptAdapter = None
        self.do_construct_prompt: bool = True             # TODO(@zyw): 不同模型组装prompt不一样，找到更好的写法
        raise NotImplementedError
    
    def fix_tokenizer(self): 
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "<|endoftext|>"
            logger.info("Add eos token: {}".format(self.tokenizer.eos_token))

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Add pad token: {}".format(self.tokenizer.pad_token))
    
    def construct_prompt(self, messages: List[ChatMessage]) -> Union[str, List[ChatMessage]]: 
        return self.prompt_adapter.construct_prompt(messages) if self.do_construct_prompt else messages
    
    def chat(self, gen_params): 
        for x in self.stream_chat(gen_params): 
            pass
        return x
    
    def stream_chat(self, gen_params): 
        if not self.use_streamer_v2: 
            yield from self.stream_chat_v1(gen_params)
        else: 
            yield from self.stream_chat_v2(gen_params)

    def stream_chat_v1(self, gen_params): 
        if isinstance(gen_params["prompt"], list): 
            gen_params["prompt"] = self.construct_prompt(gen_params["prompt"])
        
        try: 
            for output in self._generate_stream(
                self.model, 
                self.tokenizer, 
                gen_params, 
                self.device, 
                self.context_len, 
                self.stream_interval, 
            ): 
                response_dict = {
                    "text": output["text"], 
                    "error_code": 0, 
                }
                if "usage" in output: 
                    response_dict["usage"] = output["usage"]
                if "finish_reason" in output: 
                    response_dict["finish_reason"] = output["finish_reason"]
                if "logprobs" in output: 
                    response_dict["logprobs"] = output["logprobs"]
                yield response_dict

        except torch.cuda.OutOfMemoryError as e:
            response_dict = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield response_dict

        except (ValueError, RuntimeError) as e:
            response_dict = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield response_dict
    
    def stream_chat_v2(self, gen_params): 
        if isinstance(gen_params["prompt"], list):
            gen_params["prompt"] = self.construct_prompt(gen_params["prompt"])

        try:
            yield from self._generate_stream_v2(
                self.model,
                self.tokenizer,
                gen_params,
                self.device,
                self.context_len,
            )

        except torch.cuda.OutOfMemoryError as e:
            response_dict = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield response_dict

        except (ValueError, RuntimeError) as e:
            response_dict = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield response_dict

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None