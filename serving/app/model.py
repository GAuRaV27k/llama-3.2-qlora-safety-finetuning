import time
import threading

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_MODEL_ID = "GAuRaV27k/llama-3.2-qlora-safety-classifier"

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def get_model():
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        with _model_lock:
            if _model is None or _tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    dtype=torch.bfloat16,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(
                    base_model,
                    ADAPTER_MODEL_ID,
                )
                model.eval()

                _tokenizer = tokenizer
                _model = model

    return _model, _tokenizer


def classify_safety(text: str) -> dict:
    model, tokenizer = get_model()
    prompt = (
        "You are a safety classifier.\n"
        "Return exactly one label: Safe or Unsafe.\n\n"
        f"Input: {text}\n"
        "Label:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_length = inputs["input_ids"].shape[-1]

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    generated_tokens = outputs[0][input_token_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    tail = generated_text.lower()

    if tail.startswith("unsafe"):
        prediction = "Unsafe"
    elif tail.startswith("safe"):
        prediction = "Safe"
    else:
        prediction = "Unknown"

    return {
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2),
        "raw_output": generated_text,
    }
