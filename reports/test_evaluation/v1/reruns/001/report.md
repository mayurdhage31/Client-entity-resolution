# Frozen held-out test evaluation — SYNTHETIC DATA

Run: identical reproducibility rerun 001. Version: v1.

Model: stage3_logistic_regression; unchanged threshold 0.5; 21 observed-only features.

## Blocking

154 records / 60 entities; 11781 possible unordered pairs; 309 existing candidates; pair reduction 97.3771%.

True pairs retrieved 156 / 157; missed 1; blocking recall 99.3631%.

## Candidate classification and direct-pair recall

| Method | Precision | Candidate recall | F1 | TP / FP / TN / FN | End-to-end direct recall | Blocking / classifier misses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| rule_based | 100.0000% | 82.0513% | 90.1408% | 128 / 0 / 153 / 28 | 81.5287% | 1 / 28 |
| logistic_regression | 100.0000% | 96.7949% | 98.3713% | 151 / 0 / 153 / 5 | 96.1783% | 1 / 5 |

Candidate recall uses true duplicates among candidates. End-to-end direct recall uses ALL true duplicate pairs in the split. Blocking misses were never scored or added to candidates. Scores are uncalibrated; rules are illustrative, not banking policy.

## Connected-component clusters

61 clusters, 14 singletons, largest 4; 151 direct edges; 3 additional same-component pairs.

All-pairs precision 100.0000%; recall 98.0892%; F1 99.0354%; TP / FP / FN = 154 / 0 / 3.

Overmerged clusters: 0; fragmented entities: 1; review flags: 2 across 2 clusters.

Cluster metrics cover ALL unordered record pairs. Transitive co-membership is separate from a directly predicted edge. Registration conflicts flag human review without changing membership. Predictions are not approved record merges.

## Integrity and limitations

Freeze SHA-256: `eac4575ea8f451222320bc1d1b2674e2223a9bd961d0dae141f008e5aeb8bf1a`. 78 protected artifacts verified unchanged. Complete unit suite before evaluation: 98 passes.

Truth was joined only after label-free predictions, all test nodes, components and flags were committed. No retraining, retrieval change, threshold tuning or hierarchy-based merge occurred. Zero denominators are JSON null.

This small synthetic split represents one generator and one held-out partition. Results do not measure real-bank performance or production readiness. Connected components can amplify a single bad edge. Test results must not be used to tune this pipeline.
