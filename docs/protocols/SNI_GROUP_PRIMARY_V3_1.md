# SNI group-primary classification protocol v3.1

## Status

**EVALUATION PROTOCOL FROZEN. TRAINING AND TEST REMAIN LOCKED.**

This protocol does not turn the failed SNI v3 crop-claim gate into a pass.
It defines a different estimand whose independent unit is a source photograph,
not an annotation crop.

## Motivation

The SNI v3 manifest corrected the severe source-group imbalance found in v2:
validation now contains 898 Adrian and 253 Faruq source photographs. Its
combined crop-level gate still failed because two classes have fewer than 50
held-out crops and several classes contain many crops from one photograph.

All 15 visual classes nevertheless have at least 20 independent source groups
in validation and test. A source-group-primary experiment is therefore
statistically better aligned with the available evidence than treating every
crop as an independent observation.

## Frozen data

- Pixels and annotations: audited SNI instance-crop v1.
- Assignment: exact SNI source-balanced v3 manifests, seed 42.
- Labels: the frozen 15 visual classes in `ontology.json`.
- No resplitting, relabeling, recropping, or test inspection.
- Test remains locked until a later protocol explicitly opens it.

## Training rule reserved for the later model protocol

If training is authorized later, train sampling must use
`train_weight_group_class_equal`:

1. every visual class receives equal total sampling mass;
2. within a class, every source photograph receives equal mass;
3. only train is sampled; validation and test are evaluated exhaustively.

Checkpoint selection must use validation source-group/class Macro-F1. A
checkpoint selected by crop-level Macro-F1 is not compatible with this
protocol.

## Primary evaluation unit

For each model, probabilities are first averaged over all crops sharing:

```text
dataset x source group x visual class
```

The class with maximum averaged probability is the prediction for that unit.
The primary metric is Macro-F1 over these group-class units.

This definition permits a source photograph containing multiple defect classes
to contribute once to each observed class while preventing a dense photograph
with hundreds of crops from dominating one class.

## Metrics

Primary:

- source-group/class Macro-F1.

Required lower-tail evidence:

- source-group/class worst-class F1;
- per-class precision, recall, F1, and group-class support.

Secondary descriptive evidence:

- crop-level Macro-F1 and worst-class F1;
- group-class metrics separately for Adrian and Faruq.

Crop-level metrics must never be described as independent-sample evidence.

## Uncertainty

Confidence intervals use a dataset-stratified cluster bootstrap:

1. source photographs are sampled with replacement within each public dataset;
2. all group-class units belonging to a sampled photograph are retained;
3. the number of sampled source photographs per dataset is unchanged;
4. paired model comparisons reuse the same bootstrap draws.

The default implementation uses 2,000 draws for screening. A final locked
report must use at least 10,000 draws.

## Current gate

This document authorizes only implementation and validation of the evaluator
and generic checkpoint-selection support. It does not authorize architecture
training. Before training, a separate model protocol must freeze:

- baseline and candidate;
- checkpoint selection implementation;
- seed order;
- validation acceptance rule;
- artifact persistence.

## Evaluator

```bash
python -u -m bilinear_lmmd.analysis.sni_group_primary \
  --manifest-root /path/to/sni-classification-v3 \
  --prediction BASE=/path/to/base/predictions.csv \
  --prediction CAND=/path/to/candidate/predictions.csv \
  --output-dir /path/to/group-primary-report \
  --split val \
  --bootstrap-iterations 2000
```

The command rejects test by default. `--allow-test` is an explicit escape
hatch for a future protocol that has already authorized test evaluation; its
existence is not authorization by itself.
