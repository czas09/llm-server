




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







