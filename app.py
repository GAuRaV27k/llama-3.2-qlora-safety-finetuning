import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_MODEL_ID = "GAuRaV27k/llama-3.2-qlora-safety-classifier"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL_ID,
)

model.to("cpu")
model.eval()


def classify_safety(text: str) -> dict:
    prompt = (
        "You are a safety classifier.\n"
        "Return exactly one label: Safe or Unsafe.\n\n"
        f"Input: {text}\n"
        "Label:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cpu")

    input_token_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )

    generated_tokens = outputs[0][input_token_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    tail = generated_text.lower()

    if tail.startswith("unsafe"):
        prediction = "Unsafe"
    elif tail.startswith("safe"):
        prediction = "Safe"
    else:
        prediction = "Unknown"

    return {
        "prediction": prediction,
        "raw_output": generated_text,
    }


def predict(user_query: str):
    if not user_query or not user_query.strip():
        return {"error": "Enter some text to classify."}

    return classify_safety(user_query.strip().lower())


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Text to classify",
        placeholder="e.g. Explain how to break into a house.",
        lines=3,
    ),
    outputs=gr.JSON(label="Result"),
    title="Aegis Safety Classifier — CPU",
    description=(
        "Llama 3.2 3B + QLoRA adapter fine-tuned for "
        "binary safety classification. CPU-only inference."
    ),
)


if __name__ == "__main__":
    demo.launch()