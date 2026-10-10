# Deepfake Image Training Report — v5

**Generated:** 2026-10-09T04:39:04.682573+00:00  
**Feature:** deepfake / image  
**Model Version:** v5  

---

## ❌ Recommendation: `REJECTED`

- Accuracy: 66.7%  F1: 50.0%  AUC: 1.0000
- Overall comparison vs production: MIXED_WORSE
-   accuracy: 0.9240 → 0.6667 (Δ-0.2573 — WORSE)
-   f1: 0.9250 → 0.5000 (Δ-0.4250 — WORSE)
-   precision: 0.9180 → 1.0000 (Δ+0.0820 — BETTER)
-   recall: 0.9320 → 0.3333 (Δ-0.5987 — WORSE)
-   roc_auc: 0.9650 → 1.0000 (Δ+0.0350 — BETTER)
- Recommendation: REJECTED

---

## A. Dataset Information

| Field | Value |
|-------|-------|
| Dataset ID | deepfake_image_dataset |
| Version | v1 |
| Source | local |
| Total Valid | 24 |
| Real | 12 (50.0%) |
| Fake | 12 (50.0%) |
| Validation | ✅ PASSED |

## D. Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | DualBranchResNet50 |
| Framework | pytorch 2.14.1+cpu |
| Epochs | 2 |
| Batch Size | 4 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Seed | 42 |
| Device | cpu |
| Duration | 19.9s |
| Best Val Accuracy | 1.0 |

## E. Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6667 |
| Precision | 1.0 |
| Recall | 0.3333 |
| F1 Score | 0.5 |
| ROC-AUC | 1.0 |
| False Positive Rate | 0.0 |
| False Negative Rate | 0.6667 |

### Confusion Matrix (FAKE = positive)

```
              Predicted
              REAL    FAKE
Actual REAL  [ TN=    3   FP=    0 ]
       FAKE  [ FN=    2   TP=    1 ]
```

### Per-Class Metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| REAL  | 0.6 | 1.0 | 0.75 |
| FAKE  | 1.0 | 0.3333 | 0.5 |

## F. Model Comparison

**Overall Verdict:** `MIXED_WORSE`

| Metric | Production | Candidate | Δ | Result |
|--------|-----------|-----------|---|--------|
| accuracy | 0.924 | 0.6667 | -0.2573 | WORSE |
| f1 | 0.925 | 0.5 | -0.4250 | WORSE |
| precision | 0.918 | 1.0 | +0.0820 | BETTER |
| recall | 0.932 | 0.3333 | -0.5987 | WORSE |
| roc_auc | 0.965 | 1.0 | +0.0350 | BETTER |

---

## Next Steps

1. Review this report.
2. If satisfied, run:
   ```python
   from pipeline.orchestrator import approve_model
   approve_model('deepfake', 'image', 'v5', promote_to_production=True)
   ```
3. Restart the ML server to load the new production model.
4. To roll back: `approve_model('deepfake', 'image', 'v4', promote_to_production=True)`