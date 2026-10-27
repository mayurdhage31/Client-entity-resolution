> Historical development notes. Start with the [current README](../README.md) for setup and the finished demo. Stage-specific commands and claims below describe the project at that time.

﻿# Banking Client Entity Resolution and Hierarchy Matching

**Current status: the beginner-friendly Indian demo is the default app.** Launch with
`.\.venv\Scripts\python.exe -m streamlit run .\app.py` and open
`http://127.0.0.1:8501`. Use [APP_GUIDE.md](../APP_GUIDE.md) for the current walkthrough,
actual Indian record labels, official public sources and runtime files. The older
Stage 7 section describes the initial UI; its two-form/Apply instructions have
been replaced by searchable record dropdowns and one comparison button.
Earlier sections describe historical stages; their references to hidden
test results apply at those stages. The original frozen v1 test results remain
unchanged.

## Stage 1: synthetic dataset generation

**ALL DATA IS SYNTHETIC.** Every legal entity, client record, source system,
registration number, address assignment, and corporate relationship is generated
for this personal portfolio project. This is not real banking or customer data.
Any resemblance to real organisations or addresses is coincidental.

This stage creates data and validates it. It does not create candidate pairs,
train a matching model, predict relationships, or build a Streamlit application.

## Run on Windows

Open PowerShell in this project folder and run:

```powershell
.\.venv\Scripts\python.exe .\generate_data.py
```

Python 3.12, Faker, and NetworkX are required; the existing `.venv` already has
these packages. Environment activation is optional because the command selects
its Python executable directly. No downloads, credentials, API keys, or network
access are needed.

Paths are built with `pathlib` relative to `generate_data.py`, so spaces in this
project's folder name work and output does not depend on the terminal's working
directory. Missing output folders are created automatically. Each run overwrites
the four generated CSV files.

The seed is **42** for both a local Python random generator and Faker. Repeated
runs with the same code and dependency versions produce identical CSV contents.
Faker output can change between package versions. The initial environment uses
Python 3.12.15, Faker 40.43.0, and NetworkX 3.7.

## Generated files

| File | Contents |
| --- | --- |
| `data/raw/client_records.csv` | Observed source records only; no truth labels. |
| `data/ground_truth/record_entity_map.csv` | `record_id,true_entity_id`: the true entity for each observed record. |
| `data/ground_truth/entities.csv` | `true_entity_id,company_name,registration_id,address,city,postcode,country`: complete canonical details for 300 distinct legal entities. |
| `data/reference/parent_relationships.csv` | `child_entity_id,parent_entity_id,relationship_source`: explicitly supplied synthetic corporate relationships. |

The observed file has exactly these fields:

```text
record_id,source_system,company_name,registration_id,address,city,postcode,country
```

Record IDs are seeded random UUID-style values, such as `rec_` followed by 32 hex
characters. They do not encode entity identity, source system, or row number.
Rows are shuffled. Record IDs are join keys, not matching features.

CSV files use UTF-8 with a header row. Missing observed values are empty cells.
Keep identifiers as text when loading them, for example in a later stage:

```python
records = pandas.read_csv("data/raw/client_records.csv", dtype=str, keep_default_na=False)
```

The example assumes `import pandas` and that the terminal is in the project root.

## What the generator simulates

- Exactly 300 separate fictional legal entities with unique canonical names and
  registration numbers. Invented company brands and Faker UK-style addresses
  keep the example coherent. Cities and postcodes are sampled independently;
  they are not a verified real postal directory.
- Between 1 and 4 records per entity, including entities that appear only once.
  Records are sampled without replacement across five fictional systems:
  `ASTER_CRM`, `BIRCH_KYC`, `CEDAR_LENDING`, `DUNE_PAYMENTS`, and `ELM_TREASURY`.
- Spelling errors, mixed capitalisation, punctuation, `Ltd` versus `Limited`,
  abbreviated addresses, postcode formatting, and missing values. Registration
  IDs can also have formatting differences, a transcription error, or be missing.
- Twelve deliberately similar-name pairs that are **different legal entities**.
  Eight additional groups of three unrelated entities share an address. These
  difficult non-matches show why a similar name or identical office is not enough
  to identify a duplicate.
- Twelve corporate families, each with five separate legal entities and four
  relationships, including a parent, subsidiaries, and a grandchild. A parent and
  one subsidiary share an office in each family. The other 240 entities are
  independent.

These design counts describe the seeded generator; the script prints the actual
record counts, missing values, hierarchy depth, and validation results each run.

## Ground truth, supplied relationships, and leakage

Two source records are duplicates only when they map to the **same legal entity**.
A parent and subsidiary have different `true_entity_id` values and registration
numbers: related companies are separate legal entities, not duplicates.

`parent_relationships.csv` is **supplied reference data, not model predictions**.
Every row has `relationship_source=SYNTHETIC_CORPORATE_REGISTRY`. Its entity IDs
refer to `entities.csv`. In this synthetic hierarchy, a child has at most one
direct parent, and edges form an acyclic graph. An entity absent from both parent
and child columns is independent. Hierarchy data may still be incomplete in a
future real-world project; this file describes the deliberately generated example.

**Truth IDs must never become matching features.** Keep `true_entity_id`, canonical
entity details, record-to-entity truth mappings, and the entity IDs in the supplied
hierarchy separate from observed matching inputs. Use labels for evaluation and
reference IDs to describe relationships; do not join them into a model feature
matrix. Opaque record IDs must also remain identifiers rather than features.

**In a later stage, split by entity or corporate family before creating pairs.**
For an entity split, assign each legal entity and all of its observed records to
one partition. For the stronger corporate-family split, use connected components
of the parent hierarchy (ignoring edge direction), treating independent entities
as one-entity groups. Assign whole groups to train, validation, or test, then
create positive and negative pairs only within each partition. Never create pairs
first and randomly split the pairs: that can place the same entity or related
family in both training and evaluation. Fit learned preprocessing only on the
training partition. No splitting or pair generation is implemented in Stage 1.

## Stage 1 validation

The script raises a clear error if validation fails. It checks both the in-memory
data and the CSV files reloaded after saving:

- Exactly 300 distinct canonical entities with complete details and unique
  canonical company names and registration numbers.
- Unique, populated record IDs and exactly one valid truth mapping for every
  observed record, with no orphaned mappings and every entity represented.
- Between 1 and 4 records per entity, including singletons, valid fictional source
  systems, and no repeated entity/source combination.
- Exactly the eight allowed observed fields, keeping truth labels out, and the
  presence of missing observed values without completely empty client details.
- Valid parent and child references, no self-parenting, no duplicate relationships,
  at most one direct parent, and no hierarchy cycles. Both families and independent
  entities must be present.

Success produces `PASS` lines and actual dataset counts. Generated CSV files are
intentionally available to commit for the portfolio; `.gitignore` excludes the
virtual environment, Python/tool caches, `.env` files, and common credential files.

## Stage 2: cleaning, family splits, and candidate blocking

All Stage 2 outputs also contain **SYNTHETIC DATA ONLY**. This stage preserves the
generator and its raw, ground-truth, canonical, and reference CSVs. It does not
train a classifier, use embeddings, or build a Streamlit application.

Run from the project root in PowerShell after Stage 1 has completed:

```powershell
.\.venv\Scripts\python.exe .\prepare_candidates.py
.\.venv\Scripts\python.exe -B -m unittest -v test_prepare_candidates
```

Stage 2 uses Python's standard library and the already installed NetworkX and
RapidFuzz. Paths
are resolved relative to the script, including when the folder contains spaces.
Reusable functions in `prepare_candidates.py` cover reading, cleaning, grouping,
split assignment/reuse, blocking, pair validation, and evaluation. Importing the
module does not run the pipeline.

### Reading and cleaning

CSV input is read entirely as strings, without numeric or automatic NA inference.
This preserves leading zeros in identifiers and postcodes. Empty CSV cells are
represented by `""`; whitespace-only or punctuation-only observed values normalise
to `""`. Literal text such as `NA` is preserved as text. Malformed rows, duplicate
or missing columns, blank record IDs, and duplicate record IDs raise errors.

`cleaned_records.csv` retains the exact eight original observed fields and adds:

```text
company_name_normalized,registration_id_normalized,address_normalized,
city_normalized,postcode_normalized,country_normalized
```

Normalisation uses Unicode NFKC compatibility normalisation, case folding,
punctuation/symbol spacing, and collapsed whitespace. Accents remain; `&` becomes
`and`. Whole-word company suffixes such as `Limited`/`Ltd`, `Incorporated`/`Inc`,
and `Corporation`/`Corp` are standardised. Address tokens such as `St`/`Street`,
`Rd`/`Road`, and `Apt`/`Flat` are standardised. Registrations and postcodes are
compacted without numeric conversion. UK country variants (`UK`, `U.K.`, `GB`,
`United Kingdom`, and `Great Britain`) become `gb`.

No spelling correction, learned preprocessing, or imputation is performed. Missing
values stay missing. **Two missing values never form a blocking key or count as
evidence of a match.** A rule is skipped whenever any required value is missing;
other rules may still find a pair using populated fields.

### Reproducible splits and leakage controls

Seed **42** is used to assign whole connected components of the supplied corporate
hierarchy, ignoring edge direction, to train/validation/test. Independent entities
each form a group. All records of an entity, and all entities of a corporate family,
stay together. The allocation targets approximately **60%/20%/20% of both records
and entities**. A deterministic greedy allocator places large groups first and
balances counts, using seeded tie ordering. Exact fractions are not always possible
without splitting a group, especially on small datasets.

`data/splits/record_splits.csv` contains:

```text
record_id,true_entity_id,split_group_id,split,split_seed,split_ratios,split_strategy
```

Group IDs are stable hashes of the family membership. This file is partition and
evaluation metadata, **never a model-feature or blocking-input table**. Subsequent
runs reuse it without rewriting it. Current records, truth mappings, family
membership, seed, ratios, and strategy must remain compatible. Incompatible saved
assignments cause an error rather than a silent re-split. For a deliberately new
experiment, explicitly archive the assignment first; do not regenerate it to
improve evaluation results.

Runtime checks enforce record, entity, and corporate-family isolation. Truth and
parent references are used only to define/validate partitions and evaluate
blocking. **Canonical company details from `entities.csv` are never loaded by
Stage 2.** Cleaning and blocking receive only observed fields; the blocker rejects
unexpected columns such as truth IDs or family metadata. No record-ID pattern,
source-system restriction, hierarchy edge, or truth label selects candidate pairs.

### Blocking rules and configuration

Candidates are generated **separately within each split**, before evaluation labels
are joined. A candidate is a pair worth comparing later, not a predicted match.
The original baseline uses the union of these exact-key rules, which remain
unchanged in the revised configuration:

| Rule label | Required nonempty normalised values |
| --- | --- |
| `country_registration` | Country and registration ID. |
| `country_postcode_name_prefix` | Country, postcode, and company-name prefix. |
| `country_city_name_token` | Country, city, and at least one shared informative company-name token. |

`BlockingConfig` defines the prefix length, minimum token length, and excluded token
list. Defaults are a prefix of the **first up to 4 alphanumeric characters** in the
normalised company name and unique name tokens of **at least 4 characters** with
at least one letter. Tokens exclude generic legal suffixes and common business
words, including `ltd`, `limited`, `plc`, `services`, `holdings`, and `trading`.
The complete static list is in the configuration and copied into the JSON report.
No vocabulary or token frequencies are learned across partitions. Set these rules
before evaluating; never tune them using test data.

Both same-source and different-source pairs are allowed. IDs are used only as join
keys and to put each pair in one consistent order (`record_id_left < record_id_right`).
Self-pairs and duplicate/reversed pairs are removed. `blocking_rules` is a sorted,
pipe-separated list of **all rules** that retrieved a pair. Repeated tokens or
overlapping blocks never create duplicate rows.

### Fuzzy fallback retrieval improvement

The revised candidates union the original exact blocks with two configurable
RapidFuzz `WRatio` retrieval passes. `FallbackConfig` defaults are:

| Method/provenance label | Query field | Neighbours per query | Minimum score |
| --- | --- | ---: | ---: |
| `fuzzy_name_top_k` | Nonempty normalised company name. | 5 | 80 |
| `fuzzy_address_top_k` | Nonempty normalised address, **only when the query's company name is missing**. | 3 | 85 |

These are **initial candidate-retrieval settings, not final match thresholds**.
`enabled=False` disables both passes. The exact-only function
`generate_candidate_pairs` remains available; `generate_revised_candidate_pairs`
adds the fallback union. Neither function accepts truth, family, or canonical data.

Every query searches only other records in its own split, without requiring an
exact country or city. Candidate targets must have nonempty text for the chosen
field. Missing-name address queries can retrieve targets that have company names.
Records with a company name do not run an address query, even when their name
query retrieves nothing. Empty text is never scored.

Scores are computed on the existing normalised strings with `processor=None`.
Self-records are removed before ranking. Ranking uses full score descending and
record ID ascending for exact ties, retaining scores **at or above** the cutoff.
Already-blocked neighbours still consume top-k slots. Retrieval is directional;
the final unordered union keeps a pair selected by either endpoint. An endpoint
can therefore have more than k final neighbours. All exact and fallback provenance
is retained in the existing pipe-separated `blocking_rules` field. Candidate files
contain no truth labels, predicted matches, or forced pairs from diagnosis.

The baseline has been frozen before code changes:

- `reports/blocking_baseline/prepare_candidates.py`: original source snapshot for
  review, retaining its original script-relative paths.
- `reports/blocking_baseline/blocking_configuration.json`: original exact rules.
- `reports/blocking_baseline/blocking_report.json`: original metrics, unchanged.
- `data/processed/baseline/candidate_pairs_*.csv`: original candidates for all
  splits; test performance remains unreported.
- `reports/blocking_baseline/manifest.json`: baseline artifact hashes and hashes
  of the protected generator, raw data, truth map, hierarchy, and saved splits.

The normal run command now verifies this archive, diagnoses **training misses
only**, then evaluates the one supplied revised configuration once on training
and validation. It never silently replaces a baseline, regenerates saved splits,
searches settings, diagnoses validation/test misses, or evaluates test labels.
An explicit later rerun repeats the same fixed workflow. If protected inputs or
baseline files change, the command fails rather than making an invalid comparison.

The training diagnosis records normalised field states and exact-rule failure
reasons for each baseline miss. In the original 26 missed training pairs, 21 have
at least one missing registration ID, seven have a missing country, five have a
missing company name, and 14 have equal nonempty addresses. These counts overlap;
they describe failed requirements rather than independent causal categories.
Ground truth is used to identify these diagnostic pairs, never to retrieve them.
Diagnosis reports must never be used as a candidate or model-feature input.

Recorded initial fallback run (seed 42, RapidFuzz 3.14.6), using exactly the
supplied settings with no validation-driven settings search:

| Metric | Train baseline | Train revised | Validation baseline | Validation revised |
| --- | ---: | ---: | ---: | ---: |
| Candidate pairs | 468 | 1,302 | 141 | 296 |
| Retrieved true pairs / candidate matches | 436 | 460 | 128 | 139 |
| Missed true pairs | 26 | 2 | 12 | 1 |
| Candidate non-matches | 32 | 842 | 13 | 157 |
| Blocking recall | 94.37% | 99.57% | 91.43% | 99.29% |
| Pair reduction | 99.55% | 98.74% | 98.70% | 97.28% |

The total true-pair denominators remain 462 for training and 140 for validation.
The improvement recovers 24 additional training pairs and 11 validation pairs,
while also adding 810 training non-matches and 144 validation non-matches. These
additional non-matches are retained for later modelling, not removed using truth.
Test candidates are refreshed with the same settings; test performance stays hidden.

### Stage 2 outputs

| File | Contents |
| --- | --- |
| `data/processed/cleaned_records.csv` | Original observed fields plus six normalised fields; no truth or split labels. |
| `data/splits/record_splits.csv` | Persistent family-safe partition assignments and configuration metadata. |
| `data/processed/candidate_pairs_train.csv` | Training candidate IDs and all responsible rules. |
| `data/processed/candidate_pairs_validation.csv` | Validation candidate IDs and all responsible rules. |
| `data/processed/candidate_pairs_test.csv` | Test candidates saved for later final evaluation; no labels. |
| `reports/blocking_report.json` | Configuration, partition sizes, training/validation metrics, and validation status. |
| `reports/training_missed_pairs_diagnosis.json` | Training-only baseline missed-pair summary and per-rule failure details. |
| `reports/blocking_comparison.json` | Frozen baseline versus revised training/validation counts and retrieval metrics. |

Each candidate file has `record_id_left,record_id_right,blocking_rules`, including a
header when there are zero candidates. Re-running Stage 2 refreshes the processed
files and report while preserving a compatible split assignment. All remain
synthetic portfolio artefacts and can be committed.

### Evaluation definitions and checks

The report evaluates **training and validation only**. Labels are attached after
blocking and never used to add missed pairs. For a split with `n` records:

- **Total possible pairs:** `n * (n - 1) / 2`, including both same-source and
  different-source comparisons within that split.
- **Candidate pairs:** the unique union returned by blocking.
- **Pair reduction:** `1 - candidate_pairs / total_possible_pairs`.
- **All true duplicate pairs:** sum of `k * (k - 1) / 2` over entities with `k`
  records in the split, including pairs the blocker missed.
- **True duplicate pairs retrieved:** candidates whose two records have the same
  ground-truth legal entity. Parent/subsidiary pairs are not duplicates.
- **Blocking recall:** `true_duplicate_pairs_retrieved / all_true_duplicate_pairs`.
- **Missed true pairs:** `all_true_duplicate_pairs - true_duplicate_pairs_retrieved`.
- **Candidate match/non-match counts:** retrieved true pairs and all remaining
  candidate pairs, respectively. These are ground-truth evaluation counts, not
  classifier predictions; related but distinct legal entities count as non-matches.

Zero denominators produce JSON `null` and console `undefined`, not a fabricated
perfect score. Record/entity counts are included in evaluation. Per-rule candidate
counts can overlap because one pair can be retrieved by several rules.

Test partition sizes are reported for allocation checks, but **no test candidate
count, reduction, true-pair count, or recall is calculated or reported**. The
evaluation function rejects requests for test performance. Reserve it for a later
final evaluation, after all rule/model choices are fixed.

The focused `unittest` checks cover leading zeros, explicit missing-value handling,
Unicode and suffix/address normalisation, generic-token exclusions, rule overlap,
pair ordering/deduplication, same-source pairs, observed-only blocking inputs,
persistent assignment reuse, incompatible assignment rejection, split isolation,
metric denominators, and the test-evaluation guard. Fixtures and temporary files
are synthetic and do not modify Stage 1 data.

The extended suite contains 28 tests, covering fuzzy retrieval, inclusive cutoffs,
missing text never being scored, deterministic top-k ties, union provenance,
split-local search, training-only diagnosis, frozen-baseline integrity, and exactly
one revised evaluation call per allowed split. Tests use small synthetic fixtures;
they do not evaluate the actual validation or test partitions.

Limitations: exact blocking misses some spelling errors and missing-country cases;
four-character prefixes can collide; abbreviation rules can be ambiguous; and
fixed token exclusions are a transparent baseline rather than a multilingual
name parser. Large common blocks can grow quadratically because there is no block
size cap that silently drops pairs. Split balance is approximate and relies on
the completeness of the supplied family reference. High blocking recall measures
candidate retrieval, not classification accuracy. This small synthetic dataset
does not establish performance on real banking data.

Exhaustive fuzzy comparisons are acceptable for this small demo (757 records),
but scoring every query against every other record is quadratic per split, with
additional ranking work. Large datasets need a scalable retrieval index to
shortlist neighbours before fuzzy reranking. Fallback retrieval can increase
non-match candidates substantially and still miss duplicates with insufficient
text or top-k competition. Improved candidate recall is not proof of match quality.

## Stage 3: pair features and matching baselines

Run Stage 3 against the existing Stage 2 outputs:

```powershell
.\.venv\Scripts\python.exe .\train_matcher.py
.\.venv\Scripts\python.exe -B -m unittest discover -v
```

Stage 3 preserves the generator, all existing data, saved family splits, blocking
settings, and candidate retrieval. It never invokes the generator or blocking
pipeline. Its runner uses only the existing training and validation candidate
files. Test candidate files, canonical entities, and parent relationships are not
opened. Test observed records are excluded before feature construction; entity
labels are filtered by the saved train/validation record IDs before being used.
Saved family metadata is used only to validate partition isolation.

### Symmetric observed-only features

`pair_features.py` defines the explicit, ordered allowlist `FEATURE_NAMES` of 21
numeric inputs and reusable `build_pair_features`, `build_feature_frame`, and
`predict_rule_based` functions. Record lookup happens outside the feature matrix.
The feature builder accepts only six normalised observed values per record:
company name, registration ID, address, city, postcode, and country.

| Feature group | Count | Definition |
| --- | ---: | --- |
| String similarities | 4 | RapidFuzz ratio and token-sort ratio for company name and address, divided by 100. |
| Exact agreements | 4 | Nonempty equality of registration ID, country, city, and postcode. |
| Registration disagreement | 1 | Both IDs present but different. A missing ID is not disagreement. |
| Missingness indicators | 12 | Exactly one missing and both missing, separately for each of the six fields. |

All feature values are in `[0, 1]`. Similarities and exact agreements are zero when
either required value is missing, including missing/missing. A both-missing flag
of 1 explicitly represents absence, not agreement. CSV cells stay text, preserving
leading zeros. Swapping pair endpoints gives exactly the same features.

Raw IDs, truth IDs, family IDs, split names, source systems, blocking provenance,
canonical details, and hierarchy information are excluded from the model matrix.
Both fitting and prediction reject extra or reordered feature columns. Ground
truth is joined only to create the binary target and evaluate results: 1 means
the same legal entity; 0 means different entities, including parents/subsidiaries.
No missing true pairs are added to the training candidates.

### Fixed illustrative rules and logistic regression

The rule-based baseline predicts a match when either:

1. Country and registration ID both agree on nonempty values; or
2. Country agrees on a nonempty value, company-name ratio is at least `0.95`,
   address ratio is at least `0.90`, and there is no disagreement between two
   present registration IDs.

These are illustrative portfolio rules, **not banking policy**. The second rule
allows a missing registration ID but never a missing company name or address to
contribute similarity evidence.

`train_matcher.py` provides reusable `fit_matcher`, `predict_logistic`,
`save_matcher`, and `load_matcher` functions. The scikit-learn pipeline contains
`StandardScaler` followed by `LogisticRegression(random_state=42, max_iter=1000)`.
The scaler and classifier are fitted once on **training candidate pairs only**;
validation is transformed and predicted without fitting. Other estimator defaults
are recorded in the saved configuration. No hyperparameter or threshold search is
performed. A score at or above the fixed threshold **0.5** predicts a match.
`predict_proba` outputs are described as **uncalibrated model scores**, not
validated match probabilities. Retrieval WRatio cutoffs and this prediction
threshold serve different purposes.

### Matching metrics and error analysis

Both methods are evaluated separately on training and validation candidates:

- **TP/FP/TN/FN:** candidate-pair confusion counts. A classifier false negative
  is a retrieved positive predicted as a non-match.
- **Precision:** `TP / (TP + FP)`.
- **Candidate recall:** `TP / (TP + FN)`, using retrieved positives only.
- **F1:** `2 * TP / (2 * TP + FP + FN)`.
- **End-to-end direct-pair recall:** `TP / all_true_duplicate_pairs`, including
  pairs that blocking missed in the denominator.
- **Blocking misses:** true pairs absent from the fixed candidate file. Total
  end-to-end missed pairs equal blocking misses plus classifier false negatives.

Zero denominators produce JSON `null` and console `undefined`; F1 uses its direct
count formula, so it is zero when positive errors exist but TP is zero. Candidate
match/non-match counts and predicted class counts are reported explicitly. True
negatives refer only to candidates, not all possible record pairs. These are
direct pair predictions; no clustering or transitive matches are inferred.

| Output | Contents |
| --- | --- |
| `models/matcher_pipeline.joblib` | Fitted scaler and logistic-regression pipeline. |
| `models/feature_names.json` | Ordered feature allowlist checked on reload. |
| `models/matcher_config.json` | Threshold, estimator parameters, rules, feature version, input hashes, and preserved retrieval configuration. |
| `models/package_versions.json` | Python, scikit-learn, NumPy, pandas, RapidFuzz, and joblib versions. |
| `reports/matching_metrics.json` | Training/validation class counts, both methods' metrics, and validation checks. |
| `reports/matching/validation_predictions.csv` | Every validation candidate with original/normalised observed fields, 21 features, target, rule prediction, uncalibrated logistic score, and logistic prediction. |
| `reports/matching/validation_*_false_positives.csv` | All validation false positives for each method, with the same inspection fields. |
| `reports/matching/validation_*_false_negatives.csv` | All validation classifier false negatives for each method. |
| `reports/matching/blocking_misses_train.csv` and `blocking_misses_validation.csv` | Unretrieved true pairs listed separately with observations; never fitted or scored. |
| `reports/matching/error_analysis.json` | Error-file index and counts with interpretation. |

Error CSVs retain their headers when empty. IDs, labels, and retrieval provenance
in inspection reports are metadata, never model inputs. Re-running the trainer
refreshes Stage 3 artifacts only. The runner checks saved/loaded score and label
consistency and verifies that its existing input files remain unchanged.

The Stage 3 checkpoint had **48 passing tests**: the previous 28 plus 20 Stage 3
checks for symmetry, scaled similarities, leading zeros, missingness, feature
exclusions, fixed-rule boundaries, training-only fitting, threshold behaviour,
save/load consistency, evaluation denominators, split isolation, and input
preservation. An integration test rejects any attempted read of test candidates,
canonical entities, or parent relationships, and verifies training fit counts.

Recorded Stage 3 run with the fixed rules and model threshold of 0.5:

| Split / method | Precision | Candidate recall | F1 | TP / FP / TN / FN | End-to-end direct-pair recall |
| --- | ---: | ---: | ---: | --- | ---: |
| Train / rules | 100.00% | 87.39% | 93.27% | 402 / 0 / 842 / 58 | 87.01% |
| Train / logistic regression | 99.57% | 100.00% | 99.78% | 460 / 2 / 840 / 0 | 99.57% |
| Validation / rules | 100.00% | 83.45% | 90.98% | 116 / 0 / 157 / 23 | 82.86% |
| Validation / logistic regression | 100.00% | 100.00% | 100.00% | 139 / 0 / 157 / 0 | 99.29% |

Training uses 1,302 candidates: 460 matches and 842 non-matches. Validation uses
296 candidates: 139 matches and 157 non-matches. The total true-pair denominators
are 462 and 140, respectively. Blocking misses **2 training pairs and 1 validation
pair**. Classifier misses on candidates are 58/23 for the rules (train/validation)
and 0/0 for logistic regression. Blocking misses are not folded into classifier
confusion matrices or added to training data.

All 296 validation candidates are saved with observations, features, and scores.
The rules have 23 validation false-negative examples. Neither method has validation
false positives, and logistic regression has no validation classifier false
negatives, so those example files correctly contain headers only. The separate
validation blocking-miss file contains the one unretrieved true pair. Perfect
candidate validation scores for this run do not establish performance on new data.

This remains a small **synthetic-data portfolio demonstration**. Unique generated
registration IDs and simple naming/address patterns may make discrimination much
easier than real entity resolution. Candidate selection changes the evaluated
class distribution; training scores are in-sample. Validation contains only 61
entities, so a few mistakes materially affect results. Scores are uncalibrated,
and these results do not establish production readiness or real-bank performance.
Test performance remains reserved for a future final evaluation.

## Stage 4: controlled sentence-embedding experiment

This experiment adds two features to a separate matching pipeline while preserving
the Stage 3 model/reports, raw data, splits, blocking settings, and every candidate
file. It uses the public pretrained
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
locally on **CPU**, without fine-tuning or paid APIs. Only public model metadata
and weights are downloaded; observed text is encoded locally.

### Installation and exact commands

Sentence Transformers was absent and was installed into the existing `.venv` with
uv using its CPU PyTorch backend. **26 packages were added, none upgraded or
removed**. Stage 3's scikit-learn, NumPy, pandas, RapidFuzz, joblib, and NetworkX
versions were constrained to remain unchanged. `uv pip check` confirmed that all
70 installed packages were compatible.

Key added versions: sentence-transformers **6.2.0**, torch **2.14.1+cpu**,
transformers **5.20.2**, huggingface-hub **2.3.0**, tokenizers **0.23.3**, and
safetensors **0.8.0**. The full before/after package maps, additions, exact original
installation command, and installation log are under `reports/embedding_experiment/`.
`requirements-embeddings.txt` pins all 26 additions. To reproduce installation:

```powershell
uv pip install --python .\.venv\Scripts\python.exe --torch-backend cpu --constraints .\reports\embedding_experiment\preserved_dependencies.txt -r .\requirements-embeddings.txt --cache-dir .\.cache\uv
```

From the project root, the exact check and experiment commands are:

```powershell
.\.venv\Scripts\python.exe -B .\run_embedding_experiment.py --integration-check
.\.venv\Scripts\python.exe -B -m unittest discover -v
.\.venv\Scripts\python.exe -B .\run_embedding_experiment.py
```

The real-model integration check downloads/loads the actual encoder, checks four
nonempty texts and 384-dimensional unit vectors, and verifies an exact warm-cache
round trip. The complete ordinary unit suite was run **once: 58 tests passed**.
Those tests are independent of downloads; fixture vectors exist only inside unit
tests and temporary directories. They are never substituted in the experiment.

### Frozen encoder, features, and cache

Resolved model revision: **`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**.
It is saved in `models/embedding_experiment/pretrained_revision.json`; subsequent
runs reuse this immutable revision rather than resolving a newer `main` branch.
The [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
supports revision-pinned loading and explicit CPU placement. The runner loads
downloaded safetensors locally, disables remote model code, and leaves the encoder
in evaluation mode. CPU inference uses two threads and batches of 32 texts.

`embedding_features.py` adds these symmetric inputs to the original 21 features:

- `company_name_embedding_cosine`: cosine of the two normalised company-name embeddings.
- `address_embedding_cosine`: cosine of the two normalised address embeddings.

Cosines retain their natural **[-1, 1]** range, with clipping only for floating-point
roundoff. If either text is missing, its cosine is zero. All original missingness
indicators and other features remain unchanged. Only existing normalised observed
names/addresses enter the encoder. Identifiers, truth/family IDs, canonical details,
city/country fields, and hierarchy data are not embedded.

Unique nonempty texts across the training/validation candidate endpoints are encoded
once in batches. Sharing a frozen text cache across these partitions does not fit
anything on validation data. The cache key includes the text content, model ID,
resolved model revision, a preprocessing version plus normaliser-source fingerprint,
runtime versions, embedding dimension, and normalisation setting. Each entry also
validates its stored identity and vector shape/norm, preventing stale reuse.

Downloaded weights are in `.cache/huggingface/`; vectors are in `.cache/embeddings/`.
The separate real check uses `.cache/embeddings-integration/`. These directories,
uv downloads, and pretrained weight patterns are excluded from Git. Windows may
store ordinary weight copies when symlinks are unavailable; caching still works.
A download/load failure stops the run and writes its error to
`reports/embedding_experiment/failure.json`; no invented embeddings are used.

### Fixed model and measured validation comparison

The experiment fits a **separate** `StandardScaler` + `LogisticRegression` on the
23 training features, with `random_state=42`, `max_iter=1000`, and threshold `0.5`.
No search is performed. The original Stage 3 pipeline is loaded without refitting
and must reproduce its saved validation scores/predictions. Both models use the
same 296 validation candidates: 139 matches and 157 non-matches.

| Model | Precision | Recall | F1 | TP / FP / TN / FN | End-to-end recall |
| --- | ---: | ---: | ---: | --- | ---: |
| Unchanged Stage 3 | 100.00% | 100.00% | 100.00% | 139 / 0 / 157 / 0 | 99.29% |
| 23-feature embedding experiment | 100.00% | 100.00% | 100.00% | 139 / 0 / 157 / 0 | 99.29% |

**Embeddings did not improve validation metrics in this run.** There are zero changed
binary predictions. Scores did change, with a maximum absolute difference of
0.03614; both sets remain uncalibrated model scores. `changed_predictions.csv`
therefore has a header and no rows, while the full validation CSV contains both
scores, their deltas, both predictions, observed values, and all 23 features.
The same one blocking miss remains outside both classifiers.

The saved selection rule chooses higher validation F1 and retains Stage 3 on a
tie. Consequently **Stage 3 remains selected for the next stage**, with its original
21-feature schema and threshold 0.5 in `models/next_stage_choice.json`. No test
performance is used in this decision or reported.

### Timings and artifacts

Recorded CPU wall times for this single run:

| Operation | Time / cache state |
| --- | --- |
| Actual embedding inference | **2.3912 s** for 609 unique nonempty texts; cold text cache, 0 hits / 609 misses. |
| Cache lookup / writes | 0.0467 s / 5.0710 s; embedding preparation including I/O totals **7.5089 s**. |
| Model loading | 0.4245 s; pretrained weights were already warm after the real integration check. |
| Two cosine features on all candidate pairs | 0.1073 s. |
| Experimental classifier fitting | 0.0613 s. |
| Stage 3 validation prediction | **2.238 ms** for 296 candidates. |
| Embedding-model validation prediction | **5.938 ms** for the same candidates. |

Embedding inference excludes downloads, imports/model loading, and cache I/O.
Prediction times exclude embedding preparation, feature construction, loading,
and training. Download/preparation timing is recorded separately; the comparison
run's weights were warm. Reruns can have a warm vector cache and zero encoder
inference time. Small single-call timings include validation/pipeline overhead
and are not a reliable performance benchmark.

| Location | Contents |
| --- | --- |
| `models/embedding_experiment/` | Separate fitted pipeline, ordered 23-feature schema, model/cache configuration, pinned revision, and package versions. |
| `models/next_stage_choice.json` | Selected pipeline path, feature schema, threshold, validation F1 comparison, and tie rule. |
| `reports/embedding_experiment/comparison.json` | Measured comparison, timings/cache state, model revision, versions, and checks. |
| `reports/embedding_experiment/validation_predictions.csv` | Same validation pairs with both model outputs and score/label changes. |
| `reports/embedding_experiment/changed_predictions.csv` | Only changed binary predictions; empty for this run. |
| `reports/embedding_experiment/real_model_integration.json` | Actual pretrained CPU integration result and separate cache check. |
| `reports/embedding_experiment/installation_*.json`, `installation.log` | Dependency provenance and installation changes. |
| `reports/embedding_experiment/preserved_inputs.json` | SHA-256 manifest of all 38 pre-existing data/model/report/source artifacts. |
| `reports/embedding_experiment/unit_test_results.json` | Complete offline suite: one run, 58 passes. |

Integrity checks hash all protected files, including test files as opaque bytes;
test rows do not enter feature building, prediction, or evaluation. Original Stage
3 artifacts remain byte-identical. Tests cover missing text, natural-range cosines,
feature symmetry, cache invalidation/corruption, training-only fitting, save/load
consistency, and tie selection. This synthetic validation set already had perfect
candidate F1, leaving no measured improvement here. General-purpose sentence
embeddings may not distinguish similar legal entities, and this experiment does
not establish benefits on real banking data or production readiness.

## Stage 5: duplicate clusters and a separate supplied hierarchy

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -B .\build_clusters.py
.\.venv\Scripts\python.exe -B -m unittest discover -v
```

Stage 5 loads the selected **Stage 3** pipeline from `models/next_stage_choice.json`.
The ordered feature schema must agree with `models/feature_names.json`, the saved
model configuration, `pair_features.FEATURE_NAMES`, and fitted scaler/model
metadata. The selected and saved thresholds must agree. In this run the schema
has 21 features and the saved threshold is **0.5**. Any disagreement raises a clear
error. Nothing is retrained, re-split, re-blocked, or threshold-tuned.

### Predicted duplicate graphs and transitivity

For training and validation separately, `build_clusters.py`:

1. Adds **every observed record in the split** as an undirected NetworkX graph
   node, including records with no candidates or positive predictions.
2. Builds the existing symmetric observed-only features and scores only the saved
   candidates with the unchanged selected pipeline.
3. Adds a **direct predicted match edge** for a score at or above the saved
   threshold, retaining the uncalibrated model score and all blocking provenance.
4. Uses connected components as predicted duplicate clusters, including singletons.
   Cluster IDs are `clu_` plus a SHA-256 hash of the sorted member record IDs.

The graph-building functions receive no truth labels or parent relationships.
Both split graphs, memberships, and review flags are fixed before the ground-truth
map is loaded for evaluation. Truth never creates/removes an edge, merges
components, or splits a component.

Connected components apply **transitivity**: if A-B and B-C are direct matches,
all three share a cluster even when A-C has no direct predicted match. Co-clustered
membership therefore contains both directly predicted and transitively implied
pairs. Inferred membership is not given an invented edge or score. Cluster summaries
record the number of direct edges and additional implied pairs separately.

**A predicted cluster is not an approved record merge.** One incorrect edge can
overmerge entire groups and create many incorrect co-clustered pairs.

Clusters with multiple distinct nonempty normalised registration IDs in the same
nonempty normalised country are flagged for **human review**. Missing values do
not create conflicts; IDs from different countries are considered separately.
Leading zeros remain significant. Flags never change graph edges or membership.
Transcription errors can trigger review even for a correctly clustered entity.

### Cluster evaluation definitions

These metrics cover **ALL unordered record pairs within each split**, including
non-candidates, and are kept separate from direct-pair classifier metrics and
blocking recall. A predicted positive means two records share a component; a true
positive label means they belong to the same legal entity.

- **TP:** same predicted cluster and same true entity.
- **FP:** same predicted cluster but different true entities.
- **FN:** same true entity but different predicted clusters.
- **Pairwise precision:** `TP / (TP + FP)`.
- **Pairwise recall:** `TP / (TP + FN)`.
- **Pairwise F1:** `2 * TP / (2 * TP + FP + FN)`.
- **Overmerged cluster:** a predicted component contains more than one true entity.
- **Fragmented true entity:** its records occur in more than one predicted component.

Zero denominators are JSON `null`. Counts are computed using cluster/entity
contingency tables, equivalent to enumerating all record pairs. An independent
saved-artifact audit also explicitly enumerated every training/validation pair.
Reports include record/cluster counts, singleton counts, largest size, TN, and
evaluation-only overmerge/fragmentation details. No test clusters or metrics are
created.

### Actual Stage 5 results

| Metric | Training | Validation |
| --- | ---: | ---: |
| Records | 455 | 148 |
| Predicted clusters | 178 | 61 |
| Singleton clusters | 43 | 16 |
| Largest cluster size | 7 | 4 |
| Direct predicted edges | 462 | 139 |
| Same-component pairs without a direct edge | 12 | 1 |
| All possible unordered pairs | 103,285 | 10,878 |
| Cluster TP / FP / FN | 462 / 12 / 0 | 140 / 0 / 0 |
| Cluster pairwise precision | 97.47% | 100.00% |
| Cluster pairwise recall | 100.00% | 100.00% |
| Cluster pairwise F1 | 98.72% | 100.00% |
| Overmerged clusters | 1 | 0 |
| Fragmented true entities | 0 | 0 |
| Clusters flagged for registration review | 13 | 3 |

The training model's incorrect direct edges lead to one seven-record overmerged
cluster and **12 false co-clustered pairs**. This illustrates error amplification
through transitivity. Validation has 139 direct match edges but 140 true
same-component pairs: a path recovers the one missing direct pair as membership.
The previous blocking recall and direct-pair classifier results remain unchanged;
cluster recall measures a different prediction scope. Validation's three review
flags remain visible even though its cluster evaluation is perfect.

### Two separate self-contained HTML views

Open these generated files directly in a browser; no server or CDN is needed:

- [Validation duplicate-cluster example](../outputs/views/validation_duplicate_cluster.html):
  four observed records, five direct match edges, and one additional inferred
  co-membership. Names/IDs come from observed records. Hover shows observed fields,
  edge model scores, and blocking provenance; a table distinguishes direct edges
  from inferred membership. Example selection uses predicted graph structure only.
- [Supplied reference hierarchy](../outputs/views/reference_corporate_hierarchy.html):
  60 entity endpoints and 48 explicit relationships, with parent-to-child arrows.
  It is labelled **“Supplied synthetic relationships; not inferred by the matching
  model”**. Nodes and edges come only from `parent_relationships.csv`. Canonical
  names are read solely as display labels for those endpoints; unrelated canonical
  entities are not added to the graph.

The reference view receives neither predicted clusters nor record-to-entity truth
mappings. It makes no claim that any predicted cluster was linked to a corporate
entity or hierarchy. Related entities remain separate legal entities. Reference
coverage is independent of the train/validation prediction views.

`cluster_views.py` uses NetworkX layouts and Plotly with embedded JavaScript.
Displayed text is HTML-escaped; tests include malicious display strings and check
that neither HTML file loads external script/image/style resources. These files
support local hover, pan, and zoom.

### Saved artifacts, checks, and limitations

| Location | Contents |
| --- | --- |
| `outputs/duplicate_clusters/<split>/cluster_membership.csv` | One row per observed record, with deterministic cluster ID, size, and observed fields; `<split>` is `train` or `validation`. |
| `outputs/duplicate_clusters/<split>/direct_edges.csv` | Direct positive edges only, with model scores and blocking provenance. |
| `outputs/duplicate_clusters/<split>/review_flags.csv` | Conflicting-ID evidence and explicit `membership_changed=False`. |
| `outputs/duplicate_clusters/<split>/cluster_summary.csv` | Component sizes, direct/implied pair counts, and review status. |
| `outputs/views/` | Separate duplicate prediction and supplied hierarchy HTML files. |
| `reports/clustering/cluster_evaluation.json` | All-pairs cluster metrics, overmerge/fragmentation details, schema/threshold, and view metadata. |
| `reports/clustering/preserved_inputs.json` | Hashes of all 57 pre-existing data, models, reports, and source files. |
| `reports/clustering/verification.json` | Independent exhaustive-pair, split, edge, ID, HTML, and preservation checks. |
| `reports/clustering/unit_test_results.json` | One complete unit-suite run: **79 tests passed**. |

The 21 new tests cover singleton retention, deterministic IDs, inclusive threshold
handling, transitive chains, split isolation, registration review flags, all-pairs
metrics, overmerging from one incorrect edge, fragmentation, metadata mismatches,
escaped/offline HTML, reference directionality, and prediction/evaluation data
separation. Existing model-training tests fit only temporary synthetic fixtures;
the actual saved models are never retrained by Stage 5.

All 57 protected prior artifacts remain byte-identical. Integrity checks may hash
test files as opaque bytes, while test records/labels do not enter clustering or
metrics. Re-running the entry point refreshes only Stage 5 outputs. Connected
components are a simple baseline: they can propagate bad edges, miss disconnected
duplicates, and turn uncertain pair scores into hard memberships. Review flags
can have false alarms and do not prove a merge is safe. The synthetic validation
result does not establish production readiness or successful hierarchy linking.

## Stage 6: frozen held-out test evaluation

`evaluate_test.py` evaluates the saved test candidates with the selected Stage 3
model and the existing illustrative rule baseline. It never fits a scaler/model,
regenerates candidates/splits, tunes thresholds, or changes clustering logic.
All data and results remain **synthetic**. No dependencies were added in this stage.

### Exact commands and immutable evaluation versions

These commands were run in this order from the project root:

```powershell
.\.venv\Scripts\python.exe -B .\evaluate_test.py --freeze
.\.venv\Scripts\python.exe -B .\evaluate_test.py --check
.\.venv\Scripts\python.exe -B .\evaluate_test.py
```

`--freeze` saved `reports/test_evaluation/v1/freeze_manifest.json` **before any
held-out outcomes were evaluated**. It contains:

- The selected model metadata and model SHA-256, the ordered 21-feature schema,
  the saved **0.5** threshold and inclusive `>=` comparison, and fixed rule definitions.
- Exact blocking and fuzzy fallback settings, unchanged from model-training provenance.
- The existing NetworkX connected-component method, deterministic cluster-ID rule,
  singleton retention, transitivity semantics, and registration-conflict review policy.
- SHA-256 hashes of **78 protected artifacts**: source and test files, all existing
  data/splits/candidates, models, earlier reports, and earlier outputs.
- Python and all installed package versions. Runtime package changes fail verification.

The model SHA-256 is
`cf2082d5e2eda4b4662f116db3c4475d3e0939a191f13cc6fefe5022da54a61b`.
The immutable freeze manifest SHA-256 is
`eac4575ea8f451222320bc1d1b2674e2223a9bd961d0dae141f008e5aeb8bf1a`.
A separate checksum protects the manifest. Freeze writes are exclusive; an
existing freeze is verified, never refreshed with different contents. README is
explicitly excluded because it documents results afterward. Local caches and new
Stage 6 output directories are also excluded; saved evaluation artifacts have
their own integrity commits.

`--check` ran the **complete offline unit suite once: 98 passed**, zero failures,
errors, or skips, in 11.377 seconds. Its saved proof and log are tied to the exact
freeze hash. The evaluator requires this successful gate before the real test run.
The 19 new evaluator tests use artificial temporary fixtures to check denominators,
blocking versus classifier misses, all-pairs cluster counts, singleton inclusion,
split boundaries, leading zeros, immutable artifacts, and reproducible reruns.
Existing training tests fit temporary fixture models only; saved project models
are unchanged. No network downloads occur in ordinary unit tests.

The real held-out evaluation ran **once**, after the full suite passed. The
original result directories refuse overwrites. To reproduce this already completed
evaluation, use only the explicit identical-rerun command:

```powershell
.\.venv\Scripts\python.exe -B .\evaluate_test.py --rerun
```

This checks the freeze and original output commits, writes labelled
`reruns/001/` (then `002/`, etc.) directories, and requires identical metrics and
byte-identical prediction/error/cluster CSVs. It preserves the original run and
reports any reproducibility mismatch. Rerun behavior was tested on artificial
fixtures; no second real evaluation was needed here. `--evaluation-version` can
name a new freeze if evaluator checks fail **before** outcomes are exposed; after
any original evaluation is complete, creating another freeze is rejected. Do not
use this held-out result to choose new settings.

### Prediction before truth access

Saved split metadata is projected onto record ID, split and family group solely
to select the test records and validate isolation; the embedded entity-label
column is ignored during prediction. Cleaned observations remain strings,
preserving leading zeros. Only the six existing normalized observed fields reach
the existing feature builder; only its ordered numeric allowlist reaches the model.

Before test truth is loaded, the evaluator saves label-free candidate predictions,
all **154 test record nodes**, direct edges, memberships, summaries and review
flags, then commits their hashes in `prediction_commit.json`. It reuses Stage 5's
graph, component-ID and flagging functions without modification. Truth mappings
are joined afterward to create targets and metrics. Canonical details and supplied
parent relationships never influence predictions or membership; integrity checks
only hash those files as opaque bytes. Missing values never create positive
agreement. No missed pair is injected or scored after seeing its label.

### Actual held-out results

Blocking uses **154 records / 60 true entities**, with **11,781** possible unordered
within-test pairs. The **309** saved candidates contain **156 matches and 153
non-matches**. Pair reduction is **97.3771%**. Blocking retrieves **156 / 157** true
duplicate pairs, misses **1**, and has **99.3631%** recall.

| Candidate-pair metric | Fixed rules | Selected logistic regression |
| --- | ---: | ---: |
| Precision | 100.0000% | 100.0000% |
| Recall among candidate duplicates | 82.0513% | 96.7949% |
| F1 | 90.1408% | 98.3713% |
| TP / FP / TN / FN | 128 / 0 / 153 / 28 | 151 / 0 / 153 / 5 |
| End-to-end direct-pair recall | 128 / 157 = 81.5287% | 151 / 157 = 96.1783% |
| Blocking misses | 1 | 1 |
| Classifier misses among candidates | 28 | 5 |
| Total missed direct pairs | 29 | 6 |

The rule baseline remains illustrative, not banking policy. Logistic outputs are
**uncalibrated model scores**, with the saved threshold unchanged at 0.5.

| Connected-component metric | Test result |
| --- | ---: |
| Predicted clusters | 61 |
| Singleton clusters | 14 |
| Largest cluster size | 4 |
| Direct predicted match edges | 151 |
| Additional same-component pairs without a direct edge | 3 |
| Cluster TP / FP / FN / TN | 154 / 0 / 3 / 11,624 |
| All-pairs precision | 100.0000% |
| All-pairs recall | 154 / 157 = 98.0892% |
| All-pairs F1 | 99.0354% |
| Overmerged clusters | 0 |
| Fragmented true entities | 1 |
| Registration-conflict review flags / flagged clusters | 2 / 2 |

The three additional co-clustered pairs are inferred by transitivity, not invented
direct edges. A record from one true entity remains disconnected from its other
records, leaving three cluster false-negative pairs. Review flags never change
membership, and a cluster prediction is not approval to merge records.

### Metric scopes and saved outputs

- **Blocking recall** is true duplicate candidates divided by all true duplicate
  pairs in the split. **Pair reduction** is `1 - candidates / choose(records, 2)`.
- **Candidate classification** evaluates only the 309 existing candidates.
  Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is
  `2TP/(2TP+FP+FN)` with candidate-level counts.
- **End-to-end direct-pair recall** is correct direct predicted matches divided
  by all **157** true duplicate pairs. Blocking misses and classifier misses are
  disjoint counts; a blocking miss has no classifier score.
- **Cluster pairwise metrics** use all **11,781** unordered record pairs. Shared
  component membership is a predicted positive, including transitive pairs. These
  use their own TP/FP/FN counts and must not be substituted for direct-pair metrics.
  Overmerging counts components with multiple true entities; fragmentation counts
  true entities spread over multiple components.
- Zero denominators give JSON `null`, rather than a misleading perfect score.

| Location | Contents |
| --- | --- |
| `reports/test_evaluation/v1/freeze_manifest.json`, `.sha256` | Immutable configuration, versions and protected-file hashes, saved before evaluation. |
| `reports/test_evaluation/v1/unit_test_results.json`, `unit_test_log.txt` | Successful one-run 98-test gate, bound to the freeze. |
| `reports/test_evaluation/v1/original/report.md` | Readable held-out report, definitions and limitations. |
| `reports/test_evaluation/v1/original/evaluation.json` | Separate blocking, candidate classification, direct-pair and cluster metrics, including fragmentation details. |
| `reports/test_evaluation/v1/original/prediction_commit.json` | Label-free prediction/component/flag hashes committed before truth access. |
| `reports/test_evaluation/v1/original/error_analysis.json` | Counts and relative paths for all saved error tables, relative to the run's output directory. |
| `reports/test_evaluation/v1/original/verification.json`, `artifacts.json` | Protected-artifact and blind-output checks; hashes of completed results. |
| `outputs/test_evaluation/v1/original/blind_candidate_predictions.csv` | Observed fields, 21 features, provenance, both predictions and model scores, without targets. |
| `outputs/test_evaluation/v1/original/candidate_predictions.csv` | Same predictions joined to evaluation-only targets after prediction. |
| `outputs/test_evaluation/v1/original/clusters/` | Membership for every test record, direct edges/scores/provenance, review flags, component summaries. |
| `outputs/test_evaluation/v1/original/errors/` | Separate rule/model false positives/negatives, unscored blocking misses and all-pairs cluster errors. Empty error CSVs retain headers. |
| `reports/test_evaluation/v1/audit.json` | Independent checks of saved counts and hashes, without rescoring the pipeline. |

All **78** frozen artifacts still match, including earlier training/validation
reports and outputs, raw data, saved splits/candidates, both saved models and
thresholds. An independent audit recomputed counts from the saved predictions and
explicitly enumerated the 11,781 test record pairs; it did not run another model
evaluation. No Streamlit application was added.

These results cover one small synthetic generator and one held-out split. Family
isolation reduces leakage but does not establish representative language, country,
identifier, missingness or adversarial variation. High precision on 309 candidates
does not establish real-bank false-positive rates. Connected components can
amplify incorrect edges, and review flags can raise false alarms. The measured
results do not establish real-bank performance or production readiness.

## Stage 7: initial local Streamlit app (historical UI)

The app is titled **Banking Client Entity Resolution**, with subtitle **Duplicate
detection and corporate hierarchy exploration**. It is clearly identified as a
personal portfolio project using synthetic data. It runs locally and has not been
published or deployed. No packages were installed or upgraded for this stage.

### Launch on Windows (Python 3.12)

Open PowerShell in the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open **http://127.0.0.1:8501**. Stop the server with `Ctrl+C`. The checked-in
`.streamlit/config.toml` binds to `127.0.0.1`, disables usage statistics and file
watching, and supplies the light theme. Restart Streamlit after changing app
source. To use a different local port:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py --server.port 8502
```

On Linux, use the same project files and Python 3.12 environment:

```bash
./.venv/bin/python -m streamlit run app.py
```

The existing environment already contains every runtime dependency. **Do not
upgrade it to launch the app.** `requirements.txt` pins the nine app/runtime
dependencies to the installed versions: Streamlit, pandas, NumPy, RapidFuzz,
scikit-learn, SciPy, joblib, NetworkX and Plotly. On another machine, create a
separate Python 3.12 environment and install those pins there. Keep the supplied
artifacts and project folder structure with the app; `requirements.txt` alone is
not a model/data distribution.

Torch, sentence-transformers, model weights and embedding caches are **not app
dependencies**. `requirements-embeddings.txt` remains a separate optional Stage 4
experiment environment specification. The app reads its saved comparison JSON;
it does not import the experiment scripts, load an encoder, use a paid API or
download anything.

### Five sections

1. **Overview:** explains why several client records may represent one legal
   entity. Cards, rule/model comparison and a labelled candidate confusion matrix
   read metrics from `reports/test_evaluation/v1/original/evaluation.json`.
   Candidate recall uses true duplicates among candidates; direct end-to-end
   recall uses all true duplicate pairs; cluster F1 uses all unordered pairs.
   Denominators are visible. The saved benchmark covers 154 synthetic test records;
   these values are loaded from the report, not hardcoded or recomputed.
2. **Compare two records:** two independent editable forms accept company name,
   registration ID, address, city, postcode and country. Select an optional raw
   synthetic sample to prefill each form, or enter fictional details. **Apply
   edits in each form**, then select **Score pair**. Prefilled samples are applied
   immediately. Scores reuse the existing normalizers, symmetric features and
   selected frozen Stage 3 model at its saved threshold. The result is explicitly
   an uncalibrated score. Completely empty/whitespace-only or punctuation-only
   records are rejected. This is direct pair scoring and **bypasses blocking**;
   it does not create candidates, change clusters or affect benchmark metrics.
3. **Duplicate cluster explorer:** select saved training, validation or original
   test outputs; search observed fields/record or cluster IDs, filter singletons
   or registration-review flags, and select a cluster. The table shows original
   observed fields. The Plotly graph draws saved direct edges only and retains
   singletons; captions distinguish transitive membership. Edge scores and
   blocking provenance are visible. Downloads contain only the selected members'
   original observed fields. Import CSV columns as text in spreadsheet software
   to retain leading zeros. Reference truth appears only after opting in inside
   **Evaluation only — synthetic reference truth**, never as predicted identity.
   There is no automatic merge action.
4. **Corporate hierarchy:** select a supplied synthetic corporate family. Arrows
   point parent-to-child and the relationship source is shown. The label is
   **“Supplied reference relationships—not inferred”**. Only reference relationship
   endpoints create nodes; canonical names label those endpoints. No record truth
   mapping or duplicate-cluster membership is passed to this view, and no linking
   of predicted clusters to reference corporate entities is asserted.
5. **Method and limitations:** explains synthetic generation, family-separated
   splits, exact/fuzzy blocking, observed-only features, training-only fitting and
   component clustering. The saved embedding comparison shows no validation F1
   improvement and the predefined tie rule retaining the simpler model. It also
   explains missed matches, uncalibrated scores, transitive error amplification,
   review flags and the limitations of this small synthetic dataset.

Entered form values and the last score stay only in the active session, allowing
the forms to work across Streamlit reruns. They are never written to disk, logged,
added to shared caches or appended to the synthetic data. **Clear both records**
clears the session's inputs and score. Cached objects are limited to saved public
synthetic artifacts and the read-only fitted pipeline. Feature evidence describes
observed comparisons; it does not claim causality or verified legal identity.

### Required runtime files

Keep these files at their existing relative paths. Missing or modified required
artifacts produce a readable error and stop the app; nothing is regenerated.

| Files | Purpose |
| --- | --- |
| `app.py`, `portfolio_ui/*.py`, `portfolio_ui/styles.css` | Entry point, read-only adapters, cached loaders, scoring and five screens. |
| `.streamlit/config.toml`, `requirements.txt` | Local theme/server defaults and Python 3.12 runtime pins. |
| `prepare_candidates.py`, `pair_features.py`, `train_matcher.py`, `build_clusters.py`, `cluster_views.py` | Existing frozen reusable normalizers, features, prediction/model validation and reference/schema helpers. Importing them never calls their entry points. |
| `models/matcher_pipeline.joblib`, `feature_names.json`, `matcher_config.json`, `package_versions.json`, `next_stage_choice.json` | Selected Stage 3 pipeline and its saved metadata, all under `models/`. |
| `data/raw/client_records.csv` | Observed synthetic sample records for prefilling. |
| `data/reference/parent_relationships.csv`, `data/ground_truth/entities.csv` | Supplied hierarchy and canonical display names only. |
| `data/ground_truth/record_entity_map.csv` | Evaluation-only cluster truth display, opt-in and excluded from scoring. |
| `reports/test_evaluation/v1/freeze_manifest.json`, `freeze_manifest.sha256` | Original immutable v1 configuration and checksum. |
| `reports/test_evaluation/v1/original/artifacts.json`, `evaluation.json` | Original completion hashes and saved benchmark metrics. |
| `outputs/duplicate_clusters/train/`, `outputs/duplicate_clusters/validation/` | Each directory needs `cluster_membership.csv`, `direct_edges.csv`, `review_flags.csv`, `cluster_summary.csv`. |
| `outputs/test_evaluation/v1/original/clusters/` | The same four cluster files for the original held-out test run. |
| `reports/embedding_experiment/comparison.json` | Saved validation embedding experiment; no optional encoder dependencies needed. |

All paths are relative to the project files using `pathlib`, including folders
with spaces. UI text uses Streamlit's safe text/table/widget rendering; Plotly
labels and hover text are HTML-escaped. Only the static repository-owned
stylesheet uses HTML rendering. There are no external fonts, image assets or
embedded web pages.

### Checks and frozen-artifact preservation

Run the relevant app checks with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_app -v
```

Actual result: **20 checks passed** in 55.891 seconds: ten AppTest interaction
scenarios, nine adapter/display checks, and one independent subprocess check.
They cover startup with original report values, all five navigation choices,
both form submissions, match/non-match scoring, empty inputs, clearing, sample
prefill, split/singleton/review/search filters, download availability, opt-in truth,
hierarchy selection, missing model files, modified reports, feature exclusions,
leading zeros, symmetry, missing evidence, safe graph labels and arrow direction.
The subprocess blocks imports of torch, sentence-transformers, transformers and
huggingface_hub; all five sections and actual frozen-model direct scoring still
pass. No encoder integration or expensive experiment was rerun.

A separate local-server smoke check returned HTTP 200 for both the Streamlit
health endpoint (`ok`) and app shell on `127.0.0.1`. That temporary server was
stopped after the check.

The app validates the original freeze checksum, original report/output commits,
runtime file hashes, selected feature order and saved threshold. Its new adapter
permits additive UI files and the absence of optional embedding packages without
modifying the original freeze. The older Stage 6 evaluator also checks its exact
root-source inventory, so its original command now detects the newly added app
and test files. **Do not rewrite that manifest or rerun evaluation to launch the
app.** The app reads the completed original evaluation directly.

All 78 files listed in the v1 freeze remain byte-identical. The before/after audit
also verified all **123 pre-existing source/data/model/report/output files**,
including the already saved reproducibility rerun. Nothing was retrained, tuned, re-blocked or
re-evaluated. This app is a local synthetic portfolio demonstration; it does not
establish real-bank performance or production readiness.

## Current app: beginner-friendly Indian finance demo

See [APP_GUIDE.md](../APP_GUIDE.md) for the current launch instructions and a short
walkthrough using actual dropdown labels. The default experience now uses 24
separate simulated Indian source records with verified public names: State Bank
of India, HDFC Bank Limited, ICICI Bank Limited, Axis Bank Limited, Bajaj Finance
Limited and Bajaj Finserv Limited. Names are real; records, errors, `DEMO-IN`
identifiers, addresses and six-digit PIN codes are simulated. No actual customer
data, affiliation or endorsement is implied.

The new files live under `data/india_demo/` and `outputs/india_demo/v1/`. One batch
with the unchanged frozen pipeline generated 66 candidates, 28 direct links and
8 predicted groups (3 singletons, 1 review flag). These counts are not accuracy
measurements. Truth is separate and used only for educational expectations;
predictions are not repaired to agree with those expectations. There was no
retraining, rule tuning or rerun of the original held-out evaluation.

The five pages are **Start here**, **Compare two company records**, **Explore
duplicate groups**, **Explore corporate relationships**, and **How it works and
limitations**. Each gives three short instructions at the top. Record choices
autofill read-only details; optional edits affect temporary copies only. Record
or scenario changes clear scores. Group filters select complete saved groups.
Original v1 benchmark details are in an explicitly separate expander.

The default relationship view uses the officially sourced **Bajaj Finserv Limited
→ Bajaj Finance Limited** reference, with source URL and retrieval date. It is
labelled supplied public information, not predictions or an exhaustive structure.
Official name/relationship URLs are saved in the Indian reference CSVs and listed
in the guide. Relevant checks: **31 tests passed**, including Streamlit AppTest
coverage of all five pages and data/prediction isolation. All original frozen
artifacts remain unchanged. Nothing has been published or deployed.
