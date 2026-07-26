# SNI v3.1 Jiao Swin-HSSAM fail-fast protocol

## Status

**MODEL PROTOCOL FROZEN. VALIDATION-ONLY SCREENING MAY RUN; TEST LOCKED.**

This protocol follows the SNI v3.1 group-primary evaluation protocol. It does
not reuse the failed crop-level SNI v3 gate and does not claim a numerical
reproduction of Jiao et al. (2025).

## Verified literature basis

Jiao et al. use:

1. ImageNet-pretrained Swin-T at 224 x 224;
2. S3, S4, and S5 features in a top-down HS-FPN;
3. channel screening before lateral projection;
4. a SAM block containing controlled depthwise-separable convolution and
   fully connected channel enhancement;
5. Fusion Loss combining cross-entropy and focal loss.

Their Table 7 compares Swin-T with all combinations of HS-FPN, SAM, and Fusion
Loss. The paper reports a gain from 92.84% to 96.34% average accuracy on its
proprietary dataset. That number is background evidence only; it is not an
expected SNI result.

The local reconstruction choices and discrepancies with the paper-linked code
remain disclosed in `docs/protocols/JIAO_SWIN_HSSAM_PROTOCOL.md`.

## Frozen data and estimand

- Images: audited SNI instance-crop v1.
- Assignment: exact source-balanced v3 manifests, seed 42.
- Targets: 15 visual classes.
- Train sampler: `train_weight_group_class_equal`.
- Primary selection metric: validation source-group/class Macro-F1.
- Secondary metrics: crop Macro-F1 and per-domain group-class metrics.
- Test: locked.

All probabilities are averaged once per:

```text
dataset x source photograph x visual class
```

Uncertainty uses dataset-stratified source-group cluster bootstrap.

## Models

| Code | Model | Purpose |
|---|---|---|
| S3J0 | Swin-T + GAP + CE | Same-backbone control |
| S3J1 | Swin-T + HS-FPN + SAM + Fusion Loss | Jiao full mechanism |
| S3B0 | EfficientNetV2-B0 + GAP + CE | Strong repository benchmark |

All models use 224 px, the same online augmentation, 50-epoch budget, and the
same v3 train sampler. S3J0 and S3J1 differ only in the proposed Jiao package.

## Fail-fast order

### Stage 1 - mechanism

Run seed 42 for S3J0 and S3J1 only. S3J1 passes when:

- mean group-class Macro-F1 increases;
- group-class Macro-F1 improves on the required seed count;
- mean group-class Worst-F1 decreases by no more than one point;
- group-class Macro-F1 on neither Adrian nor Faruq decreases by more than one
  point.

For one seed, the required positive count is 1/1. For three seeds it is 2/3.
If this stage fails, stop; do not train S3B0, factorial variants, extra seeds,
or test.

### Stage 2 - benchmark

Only after Stage 1 passes, train S3B0 on seed 42 and compare S3J1 against it
using the identical gate. Passing Stage 1 but failing Stage 2 means the Jiao
mechanism helps Swin-T but is not superior to the established efficient CNN.

### Later confirmation

Seeds 123 and 2026 may be added only after both Stage 1 and Stage 2 pass.
Factorial attribution and test evaluation require another explicit decision;
they are not automatically authorized by a seed-42 pass.

## Artifact persistence

Every epoch must be synchronized to a write-enabled private Hugging Face model
repository. A failed required upload stops training. Google Drive stores the
human-readable reports, not the only resumable checkpoint.

## Interpretation

- S3J0 -> S3J1 tests transfer of the full Jiao mechanism.
- S3B0 -> S3J1 tests competitiveness against the strong repository baseline.
- A positive single seed is screening evidence, not a final claim.
- A failure is retained as evidence and ends this branch under the frozen
  protocol.

Reference:

- Jiao et al. (2025), *Swin-HSSAM: A green coffee bean grading method by Swin
  transformer*, PLOS One 20(5), e0322198.
