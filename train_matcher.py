"""Stage 3: fit/evaluate SYNTHETIC pair matchers on train and validation only.

Existing cleaned data, candidates, splits, and blocking code are read-only inputs.
No test candidates, canonical entities, or parent relationships are opened here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import prepare_candidates as pc
from pair_features import (
    FEATURE_NAMES, FEATURE_VERSION, NORMALIZED_FIELDS, build_feature_frame,
    predict_rule_based, validate_feature_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATED_SPLITS = ("train", "validation")
THRESHOLD = 0.5
RANDOM_STATE = 42
MAX_ITER = 1000
DISPLAY_FIELDS = ("source_system", *pc.DETAIL_FIELDS, *NORMALIZED_FIELDS)
OBSERVED_PAIR_FIELDS = tuple(f"{field}_{side}" for side in ("left", "right") for field in DISPLAY_FIELDS)
PREDICTION_FIELDS = (
    "record_id_left", "record_id_right", "blocking_rules", *OBSERVED_PAIR_FIELDS, *FEATURE_NAMES,
    "target", "rule_based_prediction", "logistic_score", "logistic_prediction",
)
BLOCKING_MISS_FIELDS = ("record_id_left", "record_id_right", *OBSERVED_PAIR_FIELDS, "target", "status")


def fit_matcher(features: pd.DataFrame, targets: Sequence[int]) -> Pipeline:
    """Only training data goes into this function; validation has no fit path."""
    validate_feature_frame(features)
    y = np.asarray(targets)
    pc.require(y.ndim == 1 and len(y) == len(features), "Training target shape mismatch")
    pc.require(set(y.tolist()) == {0, 1}, "Training requires both match and non-match examples")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER)),
    ])
    pipeline.fit(features, y)
    return pipeline


def predict_logistic(
    pipeline: Pipeline, features: pd.DataFrame, threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uncalibrated model scores and fixed-threshold binary predictions."""
    validate_feature_frame(features)
    pc.require(0 <= threshold <= 1, "Prediction threshold must be in [0, 1]")
    pc.require(tuple(pipeline.classes_) == (0, 1), "Expected binary model classes 0 and 1")
    if features.empty:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    scores = pipeline.predict_proba(features)[:, 1]
    return scores, (scores >= threshold).astype(int)


def package_versions() -> dict:
    return {"python": sys.version.split()[0], **{
        package: version(package) for package in ("scikit-learn", "numpy", "pandas", "rapidfuzz", "joblib")
    }}


def save_matcher(pipeline: Pipeline, directory: Path, provenance: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    pc.require(tuple(pipeline.feature_names_in_) == FEATURE_NAMES, "Fitted feature order mismatch")
    configuration = {
        "data_classification": "SYNTHETIC", "stage": 3, "feature_version": FEATURE_VERSION,
        "prediction_threshold": THRESHOLD, "prediction_comparison": ">=",
        "score_interpretation": "uncalibrated model scores, not calibrated match probabilities",
        "fit_scope": "training candidate pairs only; no hyperparameter or threshold search",
        "pipeline_order": ["StandardScaler", "LogisticRegression"],
        "standard_scaler": pipeline.named_steps["scaler"].get_params(),
        "logistic_regression": pipeline.named_steps["logistic_regression"].get_params(),
        "rule_based_baseline": {
            "description": "Illustrative demo rules, not banking policy",
            "a": "equal nonempty country AND equal nonempty registration ID",
            "b": "equal nonempty country AND company_name_ratio >= 0.95 AND address_ratio >= 0.90 "
                 "AND no disagreement between two present registration IDs",
        },
        "training_provenance": provenance,
    }
    joblib.dump(pipeline, directory / "matcher_pipeline.joblib")
    pc.write_json(directory / "feature_names.json", list(FEATURE_NAMES))
    pc.write_json(directory / "matcher_config.json", configuration)
    pc.write_json(directory / "package_versions.json", package_versions())
    return configuration


def load_matcher(directory: Path) -> tuple[Pipeline, float]:
    names = json.loads((directory / "feature_names.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "matcher_config.json").read_text(encoding="utf-8"))
    pc.require(tuple(names) == FEATURE_NAMES and config["feature_version"] == FEATURE_VERSION,
               "Saved feature allowlist/order is incompatible")
    pipeline = joblib.load(directory / "matcher_pipeline.joblib")
    pc.require(tuple(pipeline.feature_names_in_) == FEATURE_NAMES, "Saved pipeline feature order mismatch")
    threshold = config["prediction_threshold"]
    pc.require(0 <= threshold <= 1, "Invalid saved threshold")
    return pipeline, threshold


def load_stage3_inputs(root: Path) -> dict:
    """Use saved split metadata without regenerating splits or candidates."""
    assignments = pc.read_csv(root / "data/splits/record_splits.csv", pc.SPLIT_FIELDS)
    pc.validate_record_ids(assignments, "Saved splits")
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        pc.require(row["split"] in pc.SPLIT_NAMES and bool(row["split_group_id"]), "Invalid split metadata")
        family_splits[row["split_group_id"]].add(row["split"])
    pc.require(all(len(values) == 1 for values in family_splits.values()), "Corporate family crosses saved splits")
    selected = [row for row in assignments if row["split"] in EVALUATED_SPLITS]
    split_ids = {split: {row["record_id"] for row in selected if row["split"] == split}
                 for split in EVALUATED_SPLITS}
    pc.require(not split_ids["train"] & split_ids["validation"], "Record crosses splits")
    rows = pc.read_csv(root / "data/processed/cleaned_records.csv", pc.CLEANED_FIELDS)
    pc.validate_record_ids(rows, "Cleaned records")
    pc.require({row["record_id"] for row in rows} == {row["record_id"] for row in assignments},
               "Cleaned records and saved splits disagree")
    allowed_ids = split_ids["train"] | split_ids["validation"]
    records = {row["record_id"]: row for row in rows if row["record_id"] in allowed_ids}
    pairs = {}
    for split in EVALUATED_SPLITS:
        pairs[split] = pc.read_csv(root / f"data/processed/candidate_pairs_{split}.csv", pc.PAIR_FIELDS)
        pc.validate_candidate_pairs(pairs[split], split_ids[split])
    return {"records": records, "pairs": pairs, "split_ids": split_ids, "assignments": selected}


def load_selected_truth(path: Path, allowed_ids: set[str]) -> dict[str, str]:
    """Filter by saved train/validation record IDs BEFORE using entity labels."""
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        pc.require(reader.fieldnames == list(pc.TRUTH_FIELDS), "Unexpected truth CSV schema")
        for row in reader:
            record_id = row["record_id"]
            if record_id not in allowed_ids:
                continue  # Test entity labels are never consumed or evaluated.
            entity_id = row["true_entity_id"]
            pc.require(isinstance(entity_id, str) and bool(entity_id.strip()), "Missing selected entity label")
            pc.require(record_id not in mapping, "Duplicate selected truth mapping")
            mapping[record_id] = entity_id
    pc.require(set(mapping) == allowed_ids, "Selected truth mappings are incomplete")
    return mapping


def build_targets(pairs: Sequence[Mapping], truth: Mapping[str, str]) -> np.ndarray:
    """Labels supply the target only, never any feature values."""
    pc.require(all(value for value in truth.values()), "Missing target entity label")
    return np.array([int(truth[row["record_id_left"]] == truth[row["record_id_right"]])
                     for row in pairs], dtype=int)


def all_true_pairs_and_blocking_misses(
    split_truth: Mapping[str, str], pairs: Sequence[Mapping],
) -> tuple[int, list[dict]]:
    by_entity: dict[str, list[str]] = defaultdict(list)
    for record_id, entity_id in split_truth.items():
        by_entity[entity_id].append(record_id)
    total = sum(math.comb(len(ids), 2) for ids in by_entity.values())
    candidates = {(row["record_id_left"], row["record_id_right"]) for row in pairs}
    misses = sorted(pair for ids in by_entity.values() for pair in combinations(sorted(ids), 2)
                    if pair not in candidates)
    return total, [{"record_id_left": left, "record_id_right": right} for left, right in misses]


def evaluate_predictions(targets: Sequence[int], predictions: Sequence[int], all_true_pairs: int) -> dict:
    y, predicted = np.asarray(targets), np.asarray(predictions)
    pc.require(y.ndim == predicted.ndim == 1 and y.shape == predicted.shape, "Prediction shape mismatch")
    pc.require(set(y.tolist()) <= {0, 1} and set(predicted.tolist()) <= {0, 1}, "Expected binary labels")
    tp = int(np.sum((y == 1) & (predicted == 1)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    pc.require(all_true_pairs >= tp + fn, "All-true-pair denominator smaller than candidate positives")
    def divide(numerator: int, denominator: int):
        return numerator / denominator if denominator else None
    result = {
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "candidate_class_counts": {"match": tp + fn, "non_match": tn + fp},
        "predicted_class_counts": {"match": tp + fp, "non_match": tn + fn},
        "blocking_misses": all_true_pairs - (tp + fn),
        "classifier_misses_on_candidates": fn,
        "all_true_duplicate_pairs": all_true_pairs,
        "end_to_end_direct_pair_recall": divide(tp, all_true_pairs),
        "end_to_end_missed_pairs": all_true_pairs - tp,
    }
    result["undefined_metrics"] = [name for name in (
        "precision", "recall", "f1", "end_to_end_direct_pair_recall"
    ) if result[name] is None]
    return result


def observed_pair_row(pair: Mapping, records: Mapping[str, Mapping]) -> dict:
    row = {key: pair[key] for key in ("record_id_left", "record_id_right")}
    for side in ("left", "right"):
        record = records[pair[f"record_id_{side}"]]
        row.update({f"{field}_{side}": record[field] for field in DISPLAY_FIELDS})
    return row


def validation_prediction_rows(pairs, records, features, targets, rule_predictions, scores, model_predictions):
    rows = []
    for index, pair in enumerate(pairs):
        row = observed_pair_row(pair, records)
        row.update({"blocking_rules": pair["blocking_rules"], **features.iloc[index].to_dict(),
                    "target": int(targets[index]), "rule_based_prediction": int(rule_predictions[index]),
                    "logistic_score": float(scores[index]), "logistic_prediction": int(model_predictions[index])})
        rows.append(row)
    return rows


def run_training(project_root: Path = PROJECT_ROOT) -> dict:
    inputs = load_stage3_inputs(project_root)
    records, pairs, split_ids = inputs["records"], inputs["pairs"], inputs["split_ids"]
    # Project away originals/IDs/source metadata BEFORE building model features.
    feature_records = {record_id: {name: row[name] for name in NORMALIZED_FIELDS}
                       for record_id, row in records.items()}
    features = {split: build_feature_frame(pairs[split], feature_records) for split in EVALUATED_SPLITS}
    truth = load_selected_truth(project_root / "data/ground_truth/record_entity_map.csv", set(records))
    entity_splits: dict[str, set[str]] = defaultdict(set)
    for row in inputs["assignments"]:
        pc.require(truth[row["record_id"]] == row["true_entity_id"], "Saved selected truth mapping has changed")
        entity_splits[truth[row["record_id"]]].add(row["split"])
    pc.require(all(len(values) == 1 for values in entity_splits.values()), "Entity crosses train/validation")
    targets = {split: build_targets(pairs[split], truth) for split in EVALUATED_SPLITS}

    used_paths = ["data/processed/cleaned_records.csv", "data/splits/record_splits.csv",
                  "data/ground_truth/record_entity_map.csv", "reports/blocking_report.json",
                  "data/processed/candidate_pairs_train.csv", "data/processed/candidate_pairs_validation.csv"]
    hashes = {path: hashlib.sha256((project_root / path).read_bytes()).hexdigest() for path in used_paths}
    blocking_report = json.loads((project_root / "reports/blocking_report.json").read_text(encoding="utf-8"))
    pipeline = fit_matcher(features["train"], targets["train"])
    pc.require(int(pipeline.named_steps["scaler"].n_samples_seen_) == len(features["train"]),
               "Scaler must be fitted on training pairs only")
    provenance = {
        "input_sha256": hashes, "training_candidate_count": len(features["train"]),
        "training_class_counts": {"match": int(targets["train"].sum()),
                                  "non_match": int((targets["train"] == 0).sum())},
        "existing_blocking_configuration": blocking_report["blocking_configuration"],
        "existing_fallback_configuration": blocking_report["fallback_configuration"],
    }
    model_dir = project_root / "models"
    configuration = save_matcher(pipeline, model_dir, provenance)
    loaded_pipeline, loaded_threshold = load_matcher(model_dir)

    report = {
        "data_classification": "SYNTHETIC", "stage": 3,
        "feature_names": list(FEATURE_NAMES), "prediction_threshold": THRESHOLD,
        "score_interpretation": configuration["score_interpretation"],
        "fit_scope": "training candidate pairs only; scaler and model fitted once",
        "undefined_metric_policy": "null for zero denominators; F1 uses 2TP/(2TP+FP+FN)",
        "test_performance": "Reserved; no test candidate file opened and no test labels used",
        "package_versions": package_versions(), "evaluation": {},
    }
    error_dir = project_root / "reports/matching"
    error_index = {"data_classification": "SYNTHETIC", "validation_errors": {}, "blocking_misses": {}}
    for split in EVALUATED_SPLITS:
        split_truth = {record_id: truth[record_id] for record_id in split_ids[split]}
        all_true, blocking_misses = all_true_pairs_and_blocking_misses(split_truth, pairs[split])
        rules = predict_rule_based(features[split])
        scores, predicted = predict_logistic(pipeline, features[split], THRESHOLD)
        restored_scores, restored_predictions = predict_logistic(loaded_pipeline, features[split], loaded_threshold)
        pc.require(np.array_equal(scores, restored_scores) and np.array_equal(predicted, restored_predictions),
                   "Saved/loaded model predictions differ")
        report["evaluation"][split] = {
            "records": len(split_truth), "entities": len(set(split_truth.values())),
            "candidate_pairs": len(pairs[split]),
            "candidate_class_counts": {"match": int(targets[split].sum()),
                                       "non_match": int((targets[split] == 0).sum())},
            "all_true_duplicate_pairs": all_true, "blocking_misses": len(blocking_misses),
            "rule_based": evaluate_predictions(targets[split], rules, all_true),
            "logistic_regression": evaluate_predictions(targets[split], predicted, all_true),
        }
        miss_rows = [{**observed_pair_row(pair, records), "target": 1,
                      "status": "blocking_miss_not_scored_or_used_for_training"} for pair in blocking_misses]
        miss_path = error_dir / f"blocking_misses_{split}.csv"
        pc.write_csv(miss_path, BLOCKING_MISS_FIELDS, miss_rows)
        error_index["blocking_misses"][split] = {"count": len(miss_rows), "file": miss_path.relative_to(project_root).as_posix()}
        if split == "validation":
            rows = validation_prediction_rows(pairs[split], records, features[split], targets[split], rules, scores, predicted)
            pc.write_csv(error_dir / "validation_predictions.csv", PREDICTION_FIELDS, rows)
            for method, column in (("rule_based", "rule_based_prediction"), ("logistic_regression", "logistic_prediction")):
                error_index["validation_errors"][method] = {}
                for kind, target, prediction in (("false_positives", 0, 1), ("false_negatives", 1, 0)):
                    examples = [row for row in rows if row["target"] == target and row[column] == prediction]
                    path = error_dir / f"validation_{method}_{kind}.csv"
                    pc.write_csv(path, PREDICTION_FIELDS, examples)
                    error_index["validation_errors"][method][kind] = {
                        "count": len(examples), "file": path.relative_to(project_root).as_posix()
                    }
    error_index["interpretation"] = (
        "Classifier false negatives are retrieved positives predicted non-match. "
        "Unretrieved true pairs are listed separately, never fitted or scored. "
        "These reports are evaluation artifacts, not feature inputs."
    )
    report["validation_checks"] = {
        "feature_allowlist": "passed", "training_only_scaler_fit": "passed",
        "save_load_prediction_consistency": "passed", "split_isolation": "passed",
    }
    pc.write_json(error_dir / "error_analysis.json", error_index)
    pc.write_json(project_root / "reports/matching_metrics.json", report)
    pc.require(all(hashlib.sha256((project_root / path).read_bytes()).hexdigest() == digest
                   for path, digest in hashes.items()), "Stage 3 changed an existing input")

    print("SYNTHETIC DATA ONLY - Stage 3 fixed rules and logistic regression")
    print(f"Features: {len(FEATURE_NAMES)}; threshold: {THRESHOLD}; training-only scaler/model fit")
    for split, evaluation in report["evaluation"].items():
        print(f"{split}: {evaluation['candidate_pairs']} candidates; classes {evaluation['candidate_class_counts']}; "
              f"all true pairs {evaluation['all_true_duplicate_pairs']}; blocking misses {evaluation['blocking_misses']}")
        for method in ("rule_based", "logistic_regression"):
            metrics = evaluation[method]
            def display(value):
                return f"{value:.4f}" if value is not None else "undefined"
            print(f"  {method}: precision={display(metrics['precision'])}, recall={display(metrics['recall'])}, "
                  f"F1={display(metrics['f1'])}, TP/FP/TN/FN="
                  f"{metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}, "
                  f"end-to-end recall={display(metrics['end_to_end_direct_pair_recall'])}")
    print("PASS: feature exclusions, split isolation, training-only fitting, save/load consistency")
    print("Saved models/, reports/matching_metrics.json, and reports/matching/ error analysis")
    print("Test performance remains hidden. Logistic outputs are uncalibrated model scores.")
    return report


if __name__ == "__main__":
    run_training()
