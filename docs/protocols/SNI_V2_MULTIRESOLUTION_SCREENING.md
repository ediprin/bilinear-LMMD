# SNI v2 GAP versus multiresolution protocol

## Question

On the audited 15-class SNI classification v2 task, does direct
multi-resolution feature fusion improve EfficientNetV2-B0 over its final-stage
GAP representation under the same imbalance-aware training recipe?

This is a validation-only representation ablation. It does not test HBP,
hierarchical experts, detection, segmentation, or open-set recognition.

## Frozen data

- image root: the 31,074 audited SNI instance crops;
- label package: `SNI_classification_manifest_v2`;
- target: the 15 `visual_label` classes;
- source-image grouped train/validation/test assignment is unchanged;
- large/medium/small labels remain metadata and are not visual targets;
- inverse-square-root sample weighting is applied only to train;
- validation retains its natural grouped distribution;
- test remains locked.

The first cross-domain direction, Adrian to Faruq, is reserved for later. It
must not be used for model selection.

## Controlled models

| Code | Encoder | Representation | Classifier |
|---|---|---|---|
| S2G | ImageNet EfficientNetV2-B0 | final-stage GAP | linear, 15 classes |
| S2MR | same | stages 1--4 projected to 128 channels, aligned and fused | linear, 15 classes |

Both models use input 224, the same stochastic train augmentation, weighted
sampler, CE with label smoothing, AdamW, cosine schedule, AMP, and 50-epoch
budget. The representation is the intended difference.

## Metrics

- Macro-F1;
- balanced accuracy;
- hard-class F1 over black variants, hole-count variants, and subtle bean
  conditions;
- bottom-three class F1;
- worst-class F1;
- per-class F1 and source-group support.

Crop count must not be presented as independent biological sample count.

## Fail-fast

Stage `screen` is locked to seed 42 and validation. S2MR passes only if:

1. Macro-F1 improves;
2. Hard-F1 improves;
3. bottom-three F1 decreases by no more than one percentage point.

If it fails, stop. If it passes, stage `confirm` runs seeds 123 and 2026.
Confirmation additionally requires Macro-F1 to improve on both seeds and
Hard-F1 to improve on at least one. Test remains locked after confirmation;
the test-opening decision requires a separate frozen protocol.

The confirmation gate is computed only from seeds 123 and 2026 so the seed 42
used to select the candidate cannot make the confirmation pass. The runner
also reports a descriptive aggregate over seeds 42, 123, and 2026, but that
three-seed aggregate does not replace the independent confirmation gate.

## Persistence

Every epoch must be uploaded to the private Hugging Face artifact repository.
Training is aborted when persistent checkpoint storage is unavailable. Local
and Drive outputs are resumable, but Drive is not used as the live image
source because that would unnecessarily slow image loading.
