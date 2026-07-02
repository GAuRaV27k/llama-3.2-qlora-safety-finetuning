# Experiments

This repository tracks multiple QLoRA experiment variants for rank/epoch comparison.

## Naming convention

Use:

`task__base-model__method__r{rank}__lr{lr}__ep{epochs}__seed{seed}`

Example:

`safetycls__llama32-3b-inst__qlora__r8__lr2e-4__ep2__seed42`

This makes sweep configuration self-describing and easy to parse in dashboards.

## Current experiments

| Experiment | LoRA Rank | Epochs | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| `rank4_epoch2` | 4 | 2 | **0.8685** | **0.8678** |
| `rank8_epoch2` | 8 | 2 | 0.8650 | 0.8642 |
| `rank16_epoch1` | 16 | 1 | 0.8545 | 0.8533 |
| `rank8_epoch1` | 8 | 1 | 0.8498 | 0.8485 |

## Artifact organization recommendation

- `models/<experiment>/` for adapter/tokenizer output
- `results/<experiment>/` for metrics and prediction-level artifacts
- Keep raw checkpoints and optimizer states out of version control
- Keep aggregate benchmark tables in docs or CI-generated reports

## Portfolio/readability recommendations

1. Keep only top-N best runs in main branch.
2. Archive full sweeps in releases or cloud storage.
3. Publish best adapter with a dedicated Hugging Face model card.
4. Add script entrypoints (`train.py`, `evaluate.py`) for reproducibility beyond notebooks.

