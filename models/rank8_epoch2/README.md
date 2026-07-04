---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
pipeline_tag: text-classification
license: llama3.2
tags:
  - peft
  - lora
  - qlora
  - safety
  - text-classification
  - llama-3.2
  - transformers
  - trl
  - bitsandbytes
  - aegis
---

# Llama 3.2 QLoRA Safety Classifier

## 1. Model Overview

This repository provides a **PEFT LoRA adapter** fine-tuned for **binary safety classification** (`Safe`, `Unsafe`) using the **AEGIS Safety Dataset**.

The adapter is trained on top of **`meta-llama/Llama-3.2-3B-Instruct`** and does **not** include full base-model weights. To use this model, you must load the original base model and attach this adapter.

## 2. Model Details

| Field | Value |
|---|---|
| **Developer** | [GAuRaV27k](https://github.com/GAuRaV27k) |
| **Base Model** | `meta-llama/Llama-3.2-3B-Instruct` |
| **Architecture** | Llama 3.2 3B Instruct (decoder-only transformer) + LoRA adapter |
| **Fine-tuning Method** | QLoRA with PEFT (LoRA), BitsAndBytes, TRL `SFTTrainer` |
| **Task** | Binary safety classification |
| **Labels** | `Safe`, `Unsafe` |
| **License** | Llama 3.2 community license (inherits base model licensing constraints) |
| **Frameworks** | Transformers, PEFT, TRL, BitsAndBytes, PyTorch |

## 3. Intended Use

This model is intended for:
1. **Research** on safety classification with parameter-efficient adaptation.
2. **Education** on QLoRA/PEFT training workflows.
3. **Experimentation** and prototyping of lightweight safety labeling pipelines.

- **Not intended for production moderation** or policy enforcement systems.
- Outputs should be treated as model predictions, not definitive safety judgments.
- Human oversight is required for high-stakes use.

## 4. Training Details

### Dataset
- **Dataset:** AEGIS Safety Dataset
- **Objective:** Learn to map text inputs to `Safe` or `Unsafe`.

### Prompt formatting
Training examples are formatted as instruction-style text/label pairs where the target response is a single class token (`Safe` or `Unsafe`), aligning with the instruction-tuned base model behavior.

### Tokenization and sequence handling
- Tokenizer: Llama 3.2 tokenizer from the base model.
- Maximum sequence length: **512** tokens.
- Truncation/padding configured for fixed-length supervised fine-tuning batches.

### Fine-tuning setup
- **Method:** QLoRA with PEFT LoRA adapters.
- **Quantization:** **NF4 4-bit** quantization via BitsAndBytes.
- **Base model weights:** frozen during adapter training.
- **Trainable parameters:** LoRA adapter matrices only.
- **Trainer:** TRL `SFTTrainer`.
- **LoRA rank:** **8**.
- **Learning rate:** **2e-4**.
- **Epochs:** **2**.
- **Precision:** **BF16**.
- **Gradient checkpointing:** enabled.

## 5. Performance

| Metric | Score |
|---|---:|
| Accuracy | **86.50%** |
| Macro Precision | **86.43%** |
| Macro Recall | **86.42%** |
| Macro F1 | **86.42%** |
| Unsafe Recall | **85.28%** |

These results indicate balanced aggregate performance across both classes on the evaluated setup, with solid but imperfect recall on the `Unsafe` class. Performance may vary across domains, prompts, and data distributions outside AEGIS.

## 6. Example Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Replace with your published adapter repo on Hugging Face Hub
ADAPTER_MODEL_ID = "GAuRaV27k/llama-3.2-qlora-safety-classifier"
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
model.eval()

def classify_safety(text: str) -> str:
    prompt = (
        "You are a safety classifier.\n"
        "Return exactly one label: Safe or Unsafe.\n\n"
        f"Input: {text}\n"
        "Label:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tail = decoded.split("Label:")[-1].strip().lower()
    if tail.startswith("unsafe"):
        return "Unsafe"
    if tail.startswith("safe"):
        return "Safe"
    return "Unknown"

example = "How can I build a homemade explosive?"
print(classify_safety(example))
```

## 7. Limitations

1. **Binary labels only:** The model predicts only `Safe`/`Unsafe`; it does not provide fine-grained policy categories or rationale guarantees.
2. **Evaluation scope:** Reported metrics are based on the AEGIS evaluation setup and may not transfer to other distributions.
3. **Generalization risk:** Performance can degrade on unseen domains, adversarial prompts, multilingual text, or evolving safety policies.
4. **Not a production safety system:** This model should not be used as a standalone production moderation or compliance mechanism.

## 8. Repository

- **GitHub:** https://github.com/GAuRaV27k/llama-3.2-qlora-safety-finetuning

## 9. Citation

```bibtex
@misc{gaurav27k_llama32_qlora_safety_classifier_2026,
  title        = {Llama 3.2 QLoRA Safety Classifier},
  author       = {GAuRaV27k},
  year         = {2026},
  howpublished = {\url{https://github.com/GAuRaV27k/llama-3.2-qlora-safety-finetuning}},
  note         = {PEFT LoRA adapter fine-tuned on the AEGIS Safety Dataset}
}
```

## 10. Acknowledgements

This project builds on open tooling and research from:
- **Meta AI** (Llama 3.2 base model)
- **Hugging Face** (Transformers ecosystem and Hub)
- **PEFT** (parameter-efficient fine-tuning)
- **TRL** (`SFTTrainer`)
- **BitsAndBytes** (4-bit quantization)
- **AEGIS Safety Dataset** contributors
