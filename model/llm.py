from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import torch

def get_llm(name, use_lora, lora_r, lora_alpha):
    print(name)
    llm_tokenizer = AutoTokenizer.from_pretrained(name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        name, 
        trust_remote_code=True,
        )

    for name, param in llm_model.named_parameters():
        param.requires_grad = False
    
    # llm_model.gradient_checkpointing_enable()

    if use_lora:
        peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )

        llm_model = get_peft_model(llm_model, peft_config)
        llm_model.print_trainable_parameters()

    return llm_tokenizer, llm_model