# Evaluation

Evaluation is run on the cleaned test split and exported as reusable artifacts.

## Outputs

For each experiment under `results/<experiment>/`:
- `evaluation_metrics.json`
- `predictions.json`
- `false_predictions/false_positive.json`
- `false_predictions/false_negative.json`

## Primary metrics

- Accuracy
- Precision (macro, weighted)
- Recall (macro, weighted)
- F1 (macro, weighted)
- Per-class precision/recall/F1/support
- Confusion matrix

## Best observed run

`rank4_epoch2`:
- Accuracy: `0.8685`
- Macro F1: `0.8678`

## Error analysis workflow

1. Inspect false positives for over-conservative `Unsafe` predictions.
2. Inspect false negatives for under-sensitive `Safe` predictions.
3. Identify recurrent lexical or contextual failure modes.
4. Feed insights back into prompt formatting or sampling strategy.

![Confusion Matrix](../images/confusion_matrix.png)

