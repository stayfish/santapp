"""Validate direct 4-bit DeepSeek-V2-Lite generation without lm-eval."""

from __future__ import annotations

import json

import accelerate
import bitsandbytes
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_PATH = "/opt/models/DeepSeek-V2-Lite"


def main() -> None:
    """Load the model directly on one GPU and print a short generation."""
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "bitsandbytes": bitsandbytes.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )
    prompt = "An attention function maps a query and key-value pairs to"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"DEEPSEEK_DIRECT_GENERATION={generated}", flush=True)


if __name__ == "__main__":
    main()
