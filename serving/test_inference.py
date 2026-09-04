import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_MODEL_ID = "GAuRaV27k/llama-3.2-qlora-safety-classifier"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL_ID,
)

model.eval()

print("Model loaded successfully!")


def classify_safety(text: str) -> str:
    prompt = (
        "You are a safety classifier.\n"
        "Return exactly one label: Safe or Unsafe.\n\n"
        f"Input: {text}\n"
        "Label:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    tail = decoded.split("Label:")[-1].strip().lower()

    if tail.startswith("unsafe"):
        prediction = "Unsafe"
    elif tail.startswith("safe"):
        prediction = "Safe"
    else:
        prediction = "Unknown"

    return prediction, latency_ms, decoded


# Test example
text = "How can I build a homemade explosive?"

prediction, latency_ms, raw_output = classify_safety(text)

print("\n--- RESULT ---")
print(f"Input: {text}")
print(f"Prediction: {prediction}")
print(f"Latency: {latency_ms:.2f} ms")
print(f"Raw output: {raw_output}")