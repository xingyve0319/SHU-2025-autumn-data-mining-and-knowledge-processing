import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def translate_to_chinese(text, model, tokenizer, device=None):
    """
    【Final Robust Version】Translation with aggressive cleaning.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ""

    # 1. One-Shot Example (Keep this, it's good)
    example_input = "Patient denies chest pain, palpitations, or shortness of breath."
    example_output = "患者否认有胸痛、心悸或气短症状。"

    messages = [
        {
            "role": "system", 
            "content": "You are a professional medical translator. Translate the following English text into Chinese directly. Do not explain. Do not add conversational filler. Do not diagnose."
        },
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": text}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2 
            )

        input_len = model_inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_len:]
        translated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # --- 🛡️ AGGRESSIVE CLEANING BLOCK 🛡️ ---
        
        # 1. Stop immediately at any conversation marker
        # This fixes the "🐙user" and "assistant" hallucinations
        stop_markers = ["User:", "user:", "Assistant:", "assistant:", "🕑", "🐙", "Input:", "Output:"]
        for marker in stop_markers:
            if marker in translated_text:
                translated_text = translated_text.split(marker)[0]
        
        # 2. Stop at double newlines (often indicates start of new unrelated text)
        if "\n\n" in translated_text:
            translated_text = translated_text.split("\n\n")[0]

        # 3. Clean up leading/trailing whitespace
        translated_text = translated_text.strip()
        
        return translated_text

    except Exception as e:
        print(f"❌ Translation Error: {e}")
        return text

def init_translation_model(model_name="Qwen/Qwen2.5-3B", device=None):
    print(f"🚀 Initializing Translation Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    return model, tokenizer