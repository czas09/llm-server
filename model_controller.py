import json
import os
import sys
from typing import List, Optional

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

if sys.version_info >= (3, 9): 
    from functools import cache
else: 
    from functools import lru_cache as cache


from config import (
    MODEL_NAME, 
    MODEL_PATH, 
    ADAPTER_MODEL_PATH, 
    QUANTIZE, 
    DEVICE, 
    DEVICE_MAP, 
    NUM_GPUS, 
    LOAD_IN_8BIT, 
    LOAD_IN_4BIT, 
    USING_PTUNING_V2, 
    CONTEXT_LEN, 
    STREAM_INTERVERL, 
    PROMPT_NAME, 
)


class BaseModelAdapter: 
    """The base and the default model adapter."""
    
    model_names = []

    def match(self, model_name) -> bool: 
        return any(m in model_name for m in self.model_names) if self.model_names else True
    
    def load_model(self, model_path: str, adapter_model: Optional[str] = None, **kwargs): 
        """ 加载本地模型文件 """
        model_path = self.default_model_path if model_path is None else model_path
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": False}
        tokenizer_kwargs.update(self.tokenizer_kwargs)

        if adapter_model is not None: 
            try: 
                tokenizer = self.tokenizer_class.from_pretrained(adapter_model, **tokenizer_kwargs)
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
        if use_ptuning_v2 and adapter_model:
            prefix_encoder_file = open(f'{adapter_model}/config.json', 'r')
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

        if adapter_model is not None:
            model = self.load_adapter_model(model, tokenizer, adapter_model, is_chatglm, config_kwargs, **kwargs)

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
    
    def load_lora_model(self, model, adapter_model, model_kwargs): 
        return PeftModel.from_pretrained(
            model, 
            adapter_model, 
            torch_dtype=model_kwargs.get("torch_dtype", torch.float16), 
        )

    def load_adapter_model(self, model, tokenizer, adapter_model, is_chatglm, model_kwargs, **kwargs):
        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        if not is_chatglm and adapter_model:
            model_vocab_size = model.get_input_embeddings().weight.size(0)
            tokenzier_vocab_size = len(tokenizer)
            logger.info(f"Vocab of the base model: {model_vocab_size}")
            logger.info(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

            if model_vocab_size != tokenzier_vocab_size:
                assert tokenzier_vocab_size > model_vocab_size
                logger.info("Resize model embeddings to fit tokenizer")
                model.resize_token_embeddings(tokenzier_vocab_size)

        if use_ptuning_v2:
            prefix_state_dict = torch.load(os.path.join(adapter_model, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            model.transformer.prefix_encoder.float()
        else:
            model = self.load_lora_model(model, adapter_model, model_kwargs)

        return model

    def post_tokenizer(self, tokenizer):
        return tokenizer

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

    @property
    def default_model_name_or_path(self):
        return "fhxk/fhllm"


# A global registry for all model adapters
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_name: str) -> BaseModelAdapter: 
    """Get a model adapter for a model name."""
    for adapter in model_adapters:
        if adapter.match(model_name):
            return adapter
    raise ValueError(f"No valid model adapter for {model_name}")


def load_model(
    model_name: str, 
    model_path: str, 
    adapter_model: Optional[str] = None, 
    quantize: Optional[int] = 16, 
    device: Optional[str] = "cuda", 
    load_in_8bit: Optional[bool] = False, 
    **kwargs
): 
    model_name = model_name.lower()

    # get model adapter
    adapter = get_model_adapter(model_name)
    model, tokenizer = adapter.load_model(
        model_path,
        adapter_model,
        device=device,
        quantize=quantize,
        load_in_8bit=load_in_8bit,
        **kwargs
    )
    return model, tokenizer


class ChatGLMModelAdapter(BaseModelAdapter): 
    """ https://github.com/THUDM/ChatGLM-6B """

    model_names = ["chatglm"]

    @property
    def model_class(self): 
        return AutoModel
    
    @property
    def default_model_name_or_path(self):
        return "THUDM/chatglm2-6b"    # TODO(@zyw)
    

class BaichuanModelAdapter(BaseModelAdapter): 
    """ https://github.com/baichuan-inc/Baichuan-13B """

    model_names = ["baichuan"]

    def load_lora_model(self, model, adapter_model, model_kwargs):
        return PeftModel.from_pretrained(model, adapter_model)
    
    @property
    def default_model_name_or_path(self):
        return "baichuan-inc/Baichuan-13B-Chat"    # TODO(@zyw)


class InternLMModelAdapter(BaseModelAdapter):
    """ https://github.com/InternLM/InternLM """

    model_names = ["internlm"]

    @property
    def default_model_name_or_path(self):
        return "internlm/internlm-chat-7b"


class QwenModelAdapter(BaseModelAdapter):
    """ https://github.com/QwenLM/Qwen-7B """

    model_names = ["qwen"]

    @property
    def default_model_name_or_path(self):
        return "Qwen/Qwen-7B-Chat"
    

class XverseModelAdapter(BaseModelAdapter):
    """ https://github.com/xverse-ai/XVERSE-13B """

    model_names = ["xverse"]

    @property
    def default_model_name_or_path(self):
        return "xverse/XVERSE-13B-Chat"


register_model_adapter(ChatGLMModelAdapter)
register_model_adapter(BaichuanModelAdapter)
register_model_adapter(InternLMModelAdapter)
register_model_adapter(QwenModelAdapter)
register_model_adapter(XverseModelAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)


def get_model(): 
    """ get generate model for chat or completion. """
    from generation import ModelServer
    # from adapter import load_model

    if "chatglm" in MODEL_NAME: 
        model, tokenizer = load_chatglm_model()
    elif "baichuan" in MODEL_NAME: 
        model, tokenizer = load_baichuan_model()
    elif "qwen" in MODEL_NAME: 
        model, tokenizer = load_qwen_model()
    elif "internlm" in MODEL_NAME: 
        model, tokenizer = load_internlm_model()
    elif "xverse" in MODEL_NAME: 
        model, tokenizer = load_xverse_model()

    # model, tokenizer = load_model(
    #     MODEL_NAME,
    #     model_name_or_path=MODEL_PATH,
    #     adapter_model=ADAPTER_MODEL_PATH,
    #     quantize=QUANTIZE,
    #     device=DEVICE,
    #     device_map=DEVICE_MAP,
    #     num_gpus=NUM_GPUS,
    #     load_in_8bit=LOAD_IN_8BIT,
    #     load_in_4bit=LOAD_IN_4BIT,
    #     use_ptuning_v2=USING_PTUNING_V2,
    # )

    return ModelServer(
        model,
        tokenizer,
        DEVICE,
        model_name=MODEL_NAME,
        context_len=CONTEXT_LEN,
        stream_interval=STREAM_INTERVERL,
        prompt_name=PROMPT_NAME,
    )


MODEL = get_model()
EXCLUDE_MODELS = ["baichuan-13b", "qwen"]  # model names for special processing
