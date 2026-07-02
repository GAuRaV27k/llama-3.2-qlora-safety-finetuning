# Architecture

This project implements a supervised safety-classification pipeline over assistant responses using a quantized Llama base model and LoRA adapters.

## End-to-end flow

1. **Data ingestion** from `dataset/aegis_dataset`.
2. **Data quality checks** for split integrity, missing labels, and duplicates.
3. **Cleaning** by retaining rows with usable safety labels.
4. **Prompt construction** into instruction-style SFT examples.
5. **Tokenization** with fixed maximum sequence length.
6. **QLoRA fine-tuning** using 4-bit quantization + PEFT adapters.
7. **Evaluation** with per-class and aggregate metrics.
8. **Artifact export** for reproducible comparison.

![System Architecture](../images/architecture.png)

## Design rationale

- **Instruction-style formatting** aligns train and inference behavior.
- **Causal LM + label generation** keeps one consistent modeling interface.
- **LoRA on attention projections** captures high-impact adaptation points at low parameter cost.
- **Artifact-first workflow** makes each experiment inspectable and reproducible.

