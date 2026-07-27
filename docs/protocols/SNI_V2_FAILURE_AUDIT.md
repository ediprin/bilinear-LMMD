# SNI v2 multiresolution failure audit

## Question

Why did `S2MR` multiresolution fusion lose to the `S2G` final-stage GAP
baseline on the SNI v2 validation split?

This is a post-hoc, validation-only data diagnosis. It performs no training,
does not select a replacement model, and never reads the test manifest.

## Frozen inputs

- SNI classification manifest v2 `val.csv`;
- `S2G_seed42/predictions.csv`;
- `S2MR_seed42/predictions.csv`;
- SNI v2 `audit.json`.

Prediction rows must align exactly with the validation manifest by
execution-root-independent crop identity. Ground-truth labels must match.

## Analyses

1. Crop-level Macro-F1 and accuracy.
2. Source-group/class-level Macro-F1:
   probabilities are averaged for crops sharing `(dataset, group_id,
   visual_label)`. This prevents many crops from one source image from being
   interpreted as independent evidence.
3. Per-class sample count, source-group count, source-identity count, F1,
   GAP-only correct, S2MR-only correct, and both-wrong count.
4. Per-domain results for `adrian_detection` and `faruq_segmentation`.
5. Per-class/per-domain results.
6. Directed confusion pairs and the number of unique source groups supporting
   each confusion.
7. Statistical-readiness flags already frozen by the SNI v2 dataset audit.

## Interpretation rules

- A class with fewer than 50 validation crops or 20 validation source groups
  remains weak evidence, regardless of crop-level F1.
- A domain Macro-F1 gap of at least ten percentage points is reported as a
  large domain association, not proof of shortcut learning.
- If S2MR loses at both crop and group/class levels, the failure cannot be
  explained only by crop pseudoreplication.
- If harm is concentrated in weak classes or one source domain, the result
  points to data support/domain mismatch; it does not prove label error.
- This audit cannot establish causality and must not be used to tune on test.

## Outputs

```text
sni_v2_failure_audit.json
class_diagnostics.csv
domain_metrics.csv
class_domain_metrics.csv
confusion_pairs.csv
sample_outcomes.csv
```

The audit is expected to run on CPU in seconds and requires no image files or
GPU.
