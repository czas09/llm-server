# InternLM-Chat-7B
# InternLM-Chat-7B-v1.1
# InternLM-Chat-20B

import gc
import json
import os.path
from typing import Optional, List, Iterable

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
from utils.utils import generate_stream, generate_stream_v2, server_error_msg, prepare_logits_processor, is_partial_stop
from config import (
    MODEL_NAME, MODEL_PATH, DEVICE, CONTEXT_LEN, 
    STREAM_INTERVERL, USE_STREAMER_V2
)
from utils.constants import ErrorCode


class InternLMModelAdapter(BaseModelAdapter): 
    """
    InternLM对话模型的模型适配
    """

    def load_model_tokenizer(self, 
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
                config_kwargs["device_map"] = "auto"
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

        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        if use_ptuning_v2 and adapter_path: 
            with open(f'{adapter_path}/config.json', 'r') as f: 
                prefix_encoder_config = json.loads(f.read())

            config.pre_seq_len = prefix_encoder_config['pre_seq_len']
            config.prefix_projection = prefix_encoder_config['prefix_projection']

        # ======================================================================
        # 加载模型
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

        if adapter_path is not None: 
            model = self.load_adapter_model(model, tokenizer, adapter_path, config_kwargs, **kwargs)
        
        if device == "cuda" and num_gpus == 1 and "device_map" not in config_kwargs:
            model.to(device)

        # inference mode
        model.eval()

        return model, tokenizer
    
    def load_adapter_model(self, model, tokenizer, adapter_path, model_kwargs, **kwargs):
        use_ptuning_v2 = kwargs.get("use_ptuning_v2", False)
        resize_embeddings = kwargs.get("resize_embeddings", False)
        if adapter_path and resize_embeddings:
            model_vocab_size = model.get_input_embeddings().weight.size(0)
            tokenzier_vocab_size = len(tokenizer)
            logger.info(f"Vocab of the base model: {model_vocab_size}")
            logger.info(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

            if model_vocab_size != tokenzier_vocab_size:
                assert tokenzier_vocab_size > model_vocab_size
                logger.info("Resize model embeddings to fit tokenizer")
                model.resize_token_embeddings(tokenzier_vocab_size)

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

        return model
    
    @property
    def model_type(self): 
        return "internlm"


class InternLMPromptAdapter(BasePromptAdapter): 
    """
    InternLM对话模型的提示词适配

    参考链接：
    InternLM-Chat-7B: TODO
    InternLM-Chat-7B-v1.1: TODO
    InternLM-Chat-20B: TODO

    InternLM对话模型的提示词格式如下所示：
    <s><|User|>:{query0}<eoh>\n<|Bot|>:{response0}<eoa>\n<s><|User|>:{query1}<eoh>\n<|Bot|>:
    （其中 s = start，eoh = end-of-human，eoa = end-of-assistant）
    """

    def __init__(self): 
        self.system_prompt = ""    # InternLM对话模型没有支持系统提示词
        self.user_prompt = "<s><|User|>:{}<eoh>\n<|Bot|>:"
        self.assistant_prompt = "{}<eoa>\n"
        self.stop = {
            "strings": ["</s>", "<eoa>"],
        }
    
    def construct_prompt(self, messages: List[ChatMessage]) -> str: 
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


class InternLM(BaseChatModel): 
    """InternLM对话模型"""

    def __init__(self): 
        self.model, self.tokenizer = self._get_model_tokenizer()
        self.model_adapter: InternLMModelAdapter = self._get_model_adapter()
        self.prompt_adapter: InternLMPromptAdapter = self._get_prompt_adapter()
        self.device = DEVICE
        self.model_name = MODEL_NAME
        # self.prompt_name = 
        self.context_len: Optional[int] = CONTEXT_LEN
        self.stream_interval: Optional[int] = STREAM_INTERVERL
        self.use_streamer_v2: Optional[bool] = USE_STREAMER_V2
        self.fix_tokenizer()
    
    def _get_model_tokenizer(self): 
        return self.model_adapter.load_model_tokenizer()

    def _get_model_adapter(self) -> InternLMModelAdapter: 
        """获取模型适配"""
        internlm_model_adapter = InternLMModelAdapter()
        return internlm_model_adapter
    
    def _get_prompt_adapter(self) -> InternLMPromptAdapter: 
        """获取提示词适配"""
        internlm_prompt_adapter = InternLMPromptAdapter()
        return internlm_prompt_adapter
    
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
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield response_dict

        except (ValueError, RuntimeError) as e:
            response_dict = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield response_dict
    
    def stream_chat_v2(self, gen_params): 
        if isinstance(gen_params["prompt"], list):
            gen_params["prompt"] = self.generate_prompt(gen_params["prompt"])


        try:
            yield from generate_stream_v2(
                self.model,
                self.tokenizer,
                gen_params,
                self.device,
                self.context_len,
            )

        except torch.cuda.OutOfMemoryError as e:
            response_dict = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield response_dict

        except (ValueError, RuntimeError) as e:
            response_dict = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield response_dict
    
    @torch.inference_mode()
    def _generate_stream(
        model,
        tokenizer,
        gen_params,
        device: str,
        context_len: int,
        stream_interval: int = 2,
    ): 
        prompt = gen_params["prompt"]
        temperature = float(gen_params.get("temperature", 1.0))
        repetition_penalty = float(gen_params.get("repetition_penalty", 1.0))
        top_p = float(gen_params.get("top_p", 1.0))
        top_k = int(gen_params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(gen_params.get("max_tokens", 256))
        echo = bool(gen_params.get("echo", True))
        stop_str = gen_params.get("stop", None)

        stop_token_ids = gen_params.get("stop_token_ids", None) or []
        if tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(tokenizer.eos_token_id)

        infilling = gen_params.get("infilling", False)
        suffix_first = gen_params.get("suffix_first", False)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )

        if infilling:
            input_ids = tokenizer(prompt, suffix_first=suffix_first).input_ids
            stop_token_ids.append(tokenizer.eot_id)
        else: 
            input_ids = tokenizer(prompt).input_ids
        
        max_src_len = context_len - max_new_tokens - 1
        input_ids = input_ids[-max_src_len:]
        output_ids = list(input_ids)
        input_echo_len = len(input_ids)

        past_key_values = None
        sent_interrupt = False
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:  # decoding
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
                past_key_values = out.past_key_values
            
            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                _, indices = torch.topk(last_token_logits, 2)
                tokens = [int(index) for index in indices.tolist()]
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                indices = torch.multinomial(probs, num_samples=2)
                tokens = [int(token) for token in indices.tolist()]

            token = tokens[0]
            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            # Yield the output tokens
            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len(prompt) if isinstance(prompt, str) else 0
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True, 
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = is_partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = is_partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                # Prevent yielding partial stop sequence
                if not partially_stopped:
                    yield {
                        "text": output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": None,
                    }

            if stopped:
                break

        # Finish stream event, which contains finish reason
        if i == max_new_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }

        # Clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()