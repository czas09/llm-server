from vllm import LLM, SamplingParams
import time


def load_model_and_tokenizer_vllm(model_dir: str = None, gpu_memory_utilization: float = 0.9): 
    llm = LLM(
        model=model_dir, 
        tokenizer=model_dir, 
        trust_remote_code=True, 
        gpu_memory_utilization=gpu_memory_utilization, 
        # tensor_parallel_size=2, 
        dtype="half", 
        tensor_parallel_size=1, 
        # block_size=16
    )
    return llm


def batch_chat_vllm(llm, prompts, stop_words, temperature, top_p): 

    chat_inputs = []
    for prompt in prompts: 
        prompt = "<reserved_102>{}<reserved_103>".format(prompt)
        chat_inputs.append(prompt)
    
    # print(chat_inputs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p, 
        max_tokens=1024, 
        stop=stop_words
    )

    outputs = llm.generate(chat_inputs, sampling_params)

    return outputs


if __name__ == '__main__': 

    model_dir = "/IAOdata/models/chatglm2-6b-20230625"
    llm = load_model_and_tokenizer_vllm(model_dir, gpu_memory_utilization=0.3)
    
    prompts = [
        "请介绍一下南京美食", 
        "你好介绍一下你自己吧", 
        "介绍一下奥本海默"
    ]
    stop_words = ["鸭血", "原子弹"]

    # outputs = llm.generate(prompts, sampling_params)
    results = batch_chat_vllm(llm, prompts, stop_words, temperature=0.01, top_p=0.3)
    # results = batch_stream_generate(model, tokenizer, inputs, stop_words, temperature=0.01, top_p=0.3)
    for output in results: 
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")