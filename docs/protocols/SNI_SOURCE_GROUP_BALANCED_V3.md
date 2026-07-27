# SNI source-group-balanced classification split v3

## Status

**DATA PREPARATION AND AUDIT ONLY. NO TRAINING OR TEST EVALUATION.**

## Research question

Can the audited SNI instance crops be partitioned into train, validation, and
test so that architecture comparisons are based on independent source
photographs rather than thousands of correlated crops from a few dense images?

This protocol was frozen after the SNI v2 failure audit found:

- 4,462 Adrian validation crops from only 8 source photographs;
- 507 Faruq validation crops from 186 source photographs;
- one Adrian photograph contributing 260--769 validation crops;
- a crop-level multiresolution Worst-F1 drop dominated by `biji_muda`, whose
  58 crops came from only four photographs.

The grouped v1 split prevented source-image leakage, but optimized crop ratios
too strongly. V3 changes only the split manifest and evaluation unit. It does
not alter pixels, bounding boxes, crop margin, labels, or the locked test
policy.

## Frozen input and labels

- Input: complete audited `SNI_instance_crop_v1`.
- Images: 31,074 existing crop files; no image is copied or re-encoded.
- Primary target: the same 15 visual classes frozen in SNI classification v2.
- Group: original source photograph (`group_id`).
- Split ratios: 70/15/15.
- Seed: 42.

## Allocation

The preparer creates 512 deterministic candidate assignments. Single-class
source photographs are stratified by visual class. Dense multi-object
photographs are stratified by their source-domain signature so that they are
distributed across all splits.

Candidate selection is lexicographic:

1. minimize feasible held-out deficits in source groups per class and dataset;
2. minimize deviations in crop counts, source-group counts, dataset balance,
   class balance, and within-class source-group concentration.

All crops from one `group_id` remain in exactly one split. The original v1
assignment is retained as metadata in `v1_generated_split`.

## Audit gate

Validation and test must each satisfy:

- at least 50 crops per visual class;
- at least 20 independent source groups per visual class;
- at least 50 independent source groups from each public dataset;
- no one source group contributes more than 25% of a class;
- zero cross-split source groups.

Failure is reported as `split_gate=FAIL`. It is not authorization to lower a
threshold after seeing model results. No model may be selected on a failed
split.

## Training-weight metadata

V3 emits three train-only columns but does not choose among them:

- `train_weight_inverse_sqrt`;
- `train_weight_group_equal`;
- `train_weight_group_class_equal`.

The last option assigns equal total sampling mass to each visual class and,
within a class, equal mass to each source photograph. A later training protocol
must freeze exactly one rule before model comparison. Validation and test are
never resampled or weighted.

## Future evaluation

Primary evaluation:

```text
source-photograph x class Macro-F1
```

Secondary evaluation:

```text
crop-level Macro-F1
```

Confidence intervals must use cluster bootstrap by source `group_id`. Individual
crops must not be bootstrapped as independent samples.

## Command

```bash
python -u -m bilinear_lmmd.data.preparation.prepare_sni_classification_v3 \
  --input-root /content/sni-instance-crops \
  --output-root /content/drive/MyDrive/sni-classification-v3 \
  --seed 42 \
  --trials 512
```

The output is a small manifest package:

```text
sni-classification-v3/
  audit.json
  ontology.json
  manifests/
    all.csv
    train.csv
    train_weighted.csv
    val.csv
    test.csv
```

## Decision rule

- `split_gate=PASS`: inspect the reported counts, then freeze one training
  weighting rule in a separate protocol.
- `split_gate=FAIL`: stop. Do not train, change thresholds, or open test.

