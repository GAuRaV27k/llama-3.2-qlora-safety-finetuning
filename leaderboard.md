# Experiment Leaderboard

This leaderboard compares all tracked QLoRA safety-classification experiments on the cleaned test split (`n=852`).

## Ranking (by Macro F1)

| Rank | Experiment | LoRA Rank | Epochs | Accuracy | Macro F1 | Safe F1 | Unsafe F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `rank4_epoch2` | 4 | 2 | **0.8685** | **0.8678** | 0.8780 | 0.8575 |
| 2 | `rank8_epoch2` | 8 | 2 | 0.8650 | 0.8642 | 0.8746 | 0.8539 |
| 3 | `rank16_epoch1` | 16 | 1 | 0.8545 | 0.8533 | 0.8664 | 0.8402 |
| 4 | `rank8_epoch1` | 8 | 1 | 0.8498 | 0.8485 | 0.8624 | 0.8346 |

## Confusion Matrix Breakdown

Format: `[[safe->safe, safe->unsafe], [unsafe->safe, unsafe->unsafe]]`

| Experiment | Confusion Matrix |
|---|---|
| `rank4_epoch2` | `[[403, 55], [57, 337]]` |
| `rank8_epoch2` | `[[401, 57], [58, 336]]` |
| `rank16_epoch1` | `[[402, 56], [68, 326]]` |
| `rank8_epoch1` | `[[401, 57], [71, 323]]` |

## Recommended Checkpoint

**Recommended for deployment candidate:** `rank4_epoch2`

Reason:
1. Highest Macro F1 and highest accuracy.
2. Best balance across safe and unsafe F1.
3. Lowest unsafe miss profile among tested runs.

## Source Artifacts

- `results/rank4_epoch2/evaluation_metrics.json`
- `results/rank8_epoch2/evaluation_metrics.json`
- `results/rank16_epoch1/evaluation_metrics.json`
- `results/rank8_epoch1/evaluation_metrics.json`

