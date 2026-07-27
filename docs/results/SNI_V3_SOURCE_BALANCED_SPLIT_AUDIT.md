# SNI v3 source-balanced split audit

**Status: SPLIT V3 CROP-CLAIM GATE FAIL; SOURCE-GROUP GATE READY FOR
PROTOCOLIZATION. NO TRAINING, TEST LOCKED.**

V3 rebuilt only the manifest assignment of the audited SNI instance crops. It
did not copy, delete, or re-encode images and did not train a model.

## Split structure

| Split | Crops | Source groups | Adrian groups | Faruq groups | Largest group |
|---|---:|---:|---:|---:|---:|
| Train | 22,170 | 5,356 | 4,139 | 1,217 | 3.79% |
| Validation | 4,740 | 1,151 | 898 | 253 | 18.46% |
| Test | 4,164 | 1,145 | 898 | 247 | 15.13% |

The main v2 defect was corrected: Adrian validation previously contained 4,462
crops from only eight source photographs. V3 contains 898 Adrian validation
groups, so architecture selection is no longer based on eight dense scenes.

## Why the frozen v3 gate failed

Every visual class has at least 20 independent source groups in validation and
test. Two classes miss the additional crop-count threshold of 50:

| Class | Validation crops/groups | Test crops/groups |
|---|---:|---:|
| `biji_muda` | 38 / 33 | 42 / 29 |
| `biji_pecah` | 46 / 29 | 47 / 32 |

Several classes also exceed the frozen rule that no one source photograph may
contribute more than 25% of a class's crops:

- validation: `biji_berkulit_tanduk` 30.86%,
  `biji_berlubang_lebih_satu` 33.64%, `biji_berlubang_satu` 40.00%,
  `biji_hitam` 25.35%, `biji_hitam_sebagian` 27.54%, and
  `kopi_gelondong` 32.89%;
- test: `biji_berlubang_lebih_satu` 36.84%,
  `biji_bertutul_tutul` 31.48%, and `biji_hitam` 27.06%.

Therefore the pre-registered combined gate correctly reported `FAIL`; its
thresholds must not be edited retroactively to turn this output into a PASS.

## Interpretation

The combined gate is stricter than the intended primary evaluation unit. If
predictions are averaged once per `source photograph x class`, crop count and
within-group crop concentration no longer determine the weight of that
photograph. Under that evaluation definition:

- every class has at least 20 independent held-out groups;
- each source dataset has far more than 50 held-out groups;
- the train/validation/test source-group ratios are close to 70/15/15;
- the test remains locked.

Thus v3 is not authorized for strong **crop-level** per-class claims, but it is
a viable input to a separately frozen **source-group-primary** protocol. That
protocol must be versioned rather than silently changing the v3 gate. It must
freeze:

1. `source photograph x class Macro-F1` as the primary selection metric;
2. crop-level Macro-F1 as secondary descriptive evidence;
3. group-class-equal train sampling;
4. cluster bootstrap by source `group_id`;
5. one-seed validation screening before any additional seeds or test.

No training was authorized or performed by this audit.

