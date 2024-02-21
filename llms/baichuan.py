# Baichuan-13B-Chat

import gc
import json
from typing import Optional, Iterable

import torch
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
)
from transformers.utils.versions import require_version
from peft import PeftModel
from loguru import logger

from llms.base import BaseChatModel, BaseModelAdapter, BasePromptAdapter
from llms.utils import build_baichuan_chat_input
from utils import prepare_logits_processor, is_partial_stop
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
            "token_ids": [195, 196],    # Baichuan 和 Baichuan2 模型的 role token ids 是一样的
        }


class Baichuan(BaseChatModel): 
    """Baichuan对话模型"""

    def __init__(self): 
        self.model_adapter: BaichuanModelAdapter = self._get_model_adapter()
        self.model, self.tokenizer = self._get_model_tokenizer()
        self.prompt_adapter: BaichuanPromptAdapter = self._get_prompt_adapter()
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
    
    def _get_model_adapter(self): 
        """获取模型适配"""
        baichuan_model_adapter = BaichuanModelAdapter()
        return baichuan_model_adapter
    
    def _get_prompt_adapter(self): 
        """获取提示词适配"""
        baichuan_prompt_adapter = BaichuanPromptAdapter()
        return baichuan_prompt_adapter
    
    @torch.inference_mode()
    def _generate_stream(
            self, 
            model,
            tokenizer,
            gen_params,
            device: str,
            context_len: int,
            stream_interval: int = 2,
    ): 
        """流式文本生成接口实现（参考 api-for-open-llm 和 fastchat 两个项目）"""
        # 这里是 FastChat 项目中自行实现的解码过程，不像 Transformers 中的 generate 支持大量的解码策略
        # 接收OpenAI格式API的入参，因此这里仅支持单条输入语句调用，不支持batch并行

        if hasattr(model, "device"): 
            device = model.device

        # ======================================================================
        # 文本生成相关参数
        # ======================================================================
        prompt = gen_params["prompt"]                                            # 输入文本
        len_prompt = len(prompt)
        temperature = float(gen_params.get("temperature", 1.0))                  # 温度，用于控制生成文本的多样性
        repetition_penalty = float(gen_params.get("repetition_penalty", 1.0))    # 重复词惩罚，默认值 1.0 表示不产生作用
        top_p = float(gen_params.get("top_p", 1.0))                              # 用于限定采样范围，按照概率分布将词表从高到低累加概率到 top_p，默认值 1.0 表示不限定范围
        top_k = int(gen_params.get("top_k", -1))                                 # 用于限定采样范围，按照概率分布将词表从高到低累计数量到 top_k，默认值 -1 表示不限定范围
        max_new_tokens = int(gen_params.get("max_tokens", 256))                  # 生成文本的最大数量
        echo = bool(gen_params.get("echo", True))                                # 将输入文本合并到生成结果中一起返回
        stop_str = gen_params.get("stop", None)                                  # 停止词，生成过程遇到这里给出的词汇就被截停
        stop_token_ids = gen_params.get("stop_token_ids", None) or []            # 与停止词类似，不过这里给出的是 token ids
        if tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(tokenizer.eos_token_id)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )

        # TODO(@zyw): 优化这里的调用逻辑
        input_ids = build_baichuan_chat_input(tokenizer, prompt, context_len, max_new_tokens)
        max_src_len = context_len - max_new_tokens - 1    # 最大输入长度
        input_ids = input_ids[-max_src_len:]              # 如果输入文本过长，就进行截断
        output_ids = list(input_ids)
        input_echo_len = len(input_ids)


        # ======================================================================
        # 执行文本生成解码的迭代过程
        # ======================================================================
        past_key_values = None    # 存储 KV cache 中间结果
        sent_interrupt = False
        finish_reason = None      # 结束原因，可能是整句话生成完毕、达到最大长度或者遇到停止词
        for i in range(max_new_tokens): 

            # ==================================================================
            # 迭代过程的第一阶段：生成当前迭代步骤的logits
            # ==================================================================
            # 模型前向计算得到的 out 是一个 CausalLMOutputWithPast 对象，在这里只包含 logits 与 past_key_values 两个值。
            # logits 是通过模型前向计算再叠加 lm head 计算得到的预测得分（prediction scores），包含了词表中每一个 token 的得分
            #    形状：[batch_size, sequence_length, vocab_size]
            # past_key_values 是迭代过程积累的一些隐藏状态（具体来说是自注意力层产生的 keys 与 values），可用于加速当前迭代的解码过程。
            #    形状：2层嵌套元组，(1) 外层元组的元素数量对应模型层数 model_layers，也就是每个元组分别存储模型中一个层的隐藏状态；(2) 内层元组包含2个元素，分别对应 key 和 value 2个张量
            #        (3) key value 张量的形状：[batch_size, num_heads, sequence_length, embed_size_per_head]
            if i == 0:  # 预填充（prefilling）阶段：第0个logits
                out = model(
                    input_ids=torch.as_tensor(
                        [input_ids], device=device
                    ), 
                    use_cache=True
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:       # 解码阶段
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
            
            # ==================================================================
            # 迭代过程的第二阶段：对当前迭代生成的logits进行处理，并采样生成下一个token
            # ==================================================================
            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if temperature < 1e-5 or top_p < 1e-8:    # 贪婪解码
                # 贪心取概率最大者对应的下标，作为下一个 token
                _, indices = torch.topk(last_token_logits, 2)
                tokens = [int(index) for index in indices.tolist()]
            else:                                     # 采样
                # 使用 softmax 将得分放缩为概率分布
                probs = torch.softmax(last_token_logits, dim=-1)
                # 使用多项式分布采样方法在概率分布中对下标进行采样，作为下一个 token
                indices = torch.multinomial(probs, num_samples=2)
                tokens = [int(token) for token in indices.tolist()]
            token = tokens[0]    # 下一个token
            output_ids.append(token)

            # ==================================================================
            # 迭代过程的第三阶段：判断迭代生成过程是否该停止
            # ==================================================================
            # 停止判断1：stop_token_ids
            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            # 以下一些逻辑的处理时机：每当迭代 stream_interval 次、到达最大长度
            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len(prompt) if isinstance(prompt, str) else 0
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                # 将 token ids 转换为对应的文本字符串
                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True, 
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )

                # 判断停止词stop，注意其中的边界条件
                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]    # 根据停止词对生成文本进行截断
                            stopped = True           # 中断迭代过程
                        else:
                            partially_stopped = is_partial_stop(output, stop_str)         # 判断当前生成结果文本是否包含停止词的一部分
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = is_partial_stop(output, each_stop)    # 判断当前生成结果文本是否包含某个停止词的一部分
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                if not partially_stopped:    # 避免出现传出文本中包含停止词的一部分的情况
                    yield {
                        "text": output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": None,     # 流式生成迭代过程中传出的是中间结果，因此不包含 finish_reason
                    }

            if stopped:
                break

        # 流式生成过程结束，因此传出结果中包含 finish_reason
        if i == max_new_tokens - 1: 
            finish_reason = "length"    # 模型生成达到最大长度
        elif stopped:
            finish_reason = "stop"      # 停止词截停（包括eos）
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

        # 在流式迭代结束后，进行一次垃圾回收
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()
    
    def _generate_stream_v2(): 
        """基于 transformers 官方提供的 TextIteratorStreamer 接口"""
        raise NotImplementedError