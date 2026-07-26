# Current project state

Snapshot date: **2026-07-27**

This is a mutable handoff snapshot. It describes what was active when the file
was last updated; it is not evidence that the active method is superior.
Agents must verify it against protocols, raw reports, and the experiment log.

## Current user direction

- Repository scope: coffee-bean classification, not the separate YOLO/detection
  project.
- Immediate task type: validation-only controlled Swin-HSSAM/Jiao screening
  on the audited SNI instance-crop dataset; SNI v2 multiresolution is stopped.
- OSR and LMMD/UDA are not currently requested.
- Avoid additional expensive training without a frozen, literature-grounded
  comparison.

## Current runnable stage

The SNI v2 GAP-versus-multiresolution screen is complete and stopped. S2MR
lost to S2G GAP on validation seed 42: Macro `-0.68`, Hard `-0.41`,
bottom-three `-6.10`, and Worst `-21.89` points. Seeds 123/2026 and test must
not be run for this protocol.

The user explicitly resumed the separate controlled Jiao Swin-HSSAM
validation screen on the audited 21-class SNI instance crops. A later
independence audit found that the current validation split contains 4,462
Adrian crops from only 8 dense source photographs. Therefore an already-running
`SJ0` versus `SJFULL` seed-42 job may finish only as an engineering screen; it
must not be treated as thesis evidence or expanded to more seeds/test until a
source-group-balanced split is frozen.

Relevant files:

- `docs/protocols/JIAO_SWIN_HSSAM_PROTOCOL.md`;
- `src/bilinear_lmmd/experiments/run_jiao_swin_hssam_screening.py`;
- `notebooks/jiao_swin_hssam_failfast_colab.ipynb`;
- `docs/results/SNI_V2_MULTIRESOLUTION_SEED42.md`.

The previous Coffee17 multistage protocol is closed:

The Coffee17 seed-42 multistage screen reported `PASS`, but the subsequent
capacity-matched control reported `FAIL`. No additional training is currently
authorized. The only active follow-up is a validation-only post-hoc audit:

| Code | Model |
|---|---|
| BE2G | EfficientNetV2-B0 + GAP baseline |
| BE2H | EfficientNetV2-B0 + HBP comparator |
| MSF0 | Fixed three-stage spatial fusion |
| MSF1 | Adaptive stage/channel multistage recalibration |
| MSFC | Uniform-stage, capacity-matched channel control |

Relevant files:

- `docs/protocols/COFFEE17_MULTISTAGE_RECALIBRATION_V1.md`
- `src/bilinear_lmmd/modeling/multistage_recalibration.py`
- `src/bilinear_lmmd/experiments/run_multistage_recalibration_screening.py`
- `notebooks/coffee17_multistage_recalibration_colab.ipynb`

Reported seed-42 validation deltas:

| Comparison | Macro-F1 | Hard-F1 | Bottom-three F1 | Worst-F1 |
|---|---:|---:|---:|---:|
| BE2G -> MSF1 | +3.66% | +3.57% | +5.69% | +6.06% |
| BE2H -> MSF1 | +1.59% | +1.28% | +1.25% | -4.20% |
| MSF0 -> MSF1 | +0.98% | +0.95% | +4.95% | +6.06% |

Capacity-control result:

| Comparison | Delta Macro | Delta Hard | Delta bottom-three | Delta Worst |
|---|---:|---:|---:|---:|
| BE2G -> MSFC | +3.65% | +4.68% | +4.78% | +0.00% |
| BE2H -> MSFC | +1.58% | +2.40% | +0.34% | -10.26% |
| MSF0 -> MSFC | +0.97% | +2.07% | +4.04% | +0.00% |
| MSFC -> MSF1 | +0.01% | -1.12% | +0.91% | +6.06% |

MSF1 failed because Hard-F1 decreased against the capacity-matched control.
Seeds 123/2026 and test remain locked. MSFC is exploratory, not final, because
it improves Macro/Hard but loses Worst-F1 against BE2H.

The no-training per-class audit is complete: 11/17 class F1 scores were
unchanged between MSFC and MSF1; MSF1 rescued two decisions and harmed two.
With only 3--8 validation images per class, the observed lower-tail movement
is a redistribution of errors rather than evidence of a clean gain. Protocol
v1 is closed; do not run further MSF seeds or test.

## Dataset snapshot

### Coffee17

- 979 original images;
- 965 after the clean grouped audit;
- fold-1: 669 train, 97 validation, 199 test;
- current MSF screen uses this dataset because it is faster and directly
  comparable with existing BE2G/BE2H checkpoints.

### SNI instance crops

- 31,074 audited crops;
- 21 shared classes;
- grouped by source image;
- no generated cross-split identity leak;
- current validation is nevertheless dominated by 4,462 Adrian crops from only
  8 dense source photographs, versus 507 Faruq crops from 186 photographs;
- a replacement split must balance source groups per dataset/class and report
  crop-level plus source-group/class metrics with cluster-aware uncertainty;
- available for engineering screens, but the current split is not adequate for
  final architecture claims.

## Important prior evidence

These are summaries only. Use
`docs/results/EXPERIMENT_MASTER_LOG.md` and the raw reports for exact values.

- EfficientNetV2-B0 was stronger than MobileNetV3 in the controlled Coffee17
  backbone comparison.
- HBP showed a positive average Coffee17 effect in some protocols but was
  seed- and backbone-sensitive and was not universally useful across datasets.
- Several candidates produced a favorable single seed and failed multi-seed
  confirmation or lower-tail criteria.
- SNI-MRENet v1 multiresolution passed its old 21-class seed-42 screen, but
  SNI v2 multiresolution failed after the target was changed to 15 visual
  classes and imbalance-aware training was applied.

## Paused or stopped work at this snapshot

The following must not be resumed automatically:

- open-set recognition;
- LMMD/UDA robustness;
- SNI ontology expert extension;
- selective residual HBP on SNI;

Other completed failures and mixed results are listed in the master log.
Their status can be revisited only with a new explicit hypothesis, not merely
to seek a positive seed.

## Persistence status

- Branch: `agent/sni-instance-crops`.
- Source code is pushed to GitHub.
- Generic per-epoch Hugging Face checkpoint persistence is implemented.
- Current notebook artifact namespace:
  `sni-jiao-hssam-v1`.
- Current artifact repository:
  `ediprin/coffee-backbone-checkpoints`.
- A write-enabled `HF_TOKEN` is required in each Colab account.

## Result status

At this snapshot:

- SNI v2 multiresolution seed-42 screening failed and is stopped;
- Jiao Swin-HSSAM `SJ0` versus `SJFULL` seed-42 may finish only as an
  engineering screen pending a source-group-balanced SNI split;
- MSF0/MSF1 seed-42 validation screening reported PASS;
- MSFC capacity control completed and MSF1 failed its causal gate;
- the no-training per-class audit completed and supported the STOP decision;
- Coffee17 test has not been opened for this experiment.
