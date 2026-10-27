"""Stage 4: one frozen-embedding experiment; preserve every Stage 3 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import embedding_features as ef
import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm


PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTION_FIELDS = (
    "record_id_left", "record_id_right", "blocking_rules", *tm.OBSERVED_PAIR_FIELDS, *ef.FEATURE_NAMES,
    "target", "stage3_score", "stage3_prediction", "embedding_score", "embedding_prediction",
    "prediction_changed", "score_delta",
)


def assert_preserved_inputs(root):
    manifest = json.loads((root / "reports/embedding_experiment/preserved_inputs.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["sha256"].items():
        pc.require(hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest,
                   f"Protected Stage 1/2/3 artifact changed: {relative}")


def fit_embedding_matcher(features, targets):
    ef.validate_feature_frame(features)
    y = np.asarray(targets)
    pc.require(y.shape == (len(features),) and set(y.tolist()) == {0, 1}, "Training needs both target classes")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    pipeline.fit(features, y)
    pc.require(int(pipeline.named_steps["scaler"].n_samples_seen_) == len(features), "Training-only scaler fit required")
    return pipeline


def predict_embedding(pipeline, features, threshold=0.5):
    ef.validate_feature_frame(features)
    pc.require(tuple(pipeline.classes_) == (0, 1) and threshold == 0.5, "Fixed binary threshold 0.5 required")
    if not len(features):
        return np.empty(0), np.empty(0, dtype=int)
    scores = pipeline.predict_proba(features)[:, 1]
    return scores, (scores >= threshold).astype(int)


def save_embedding_matcher(pipeline, directory, context, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    pc.require(tuple(pipeline.feature_names_in_) == ef.FEATURE_NAMES, "Experimental model schema mismatch")
    joblib.dump(pipeline, directory / "matcher_pipeline.joblib")
    pc.write_json(directory / "feature_names.json", list(ef.FEATURE_NAMES))
    pc.write_json(directory / "matcher_config.json", {
        "data_classification": "SYNTHETIC", "stage": 4, "threshold": 0.5,
        "cache_context": asdict(context), "scaler": pipeline.named_steps["scaler"].get_params(),
        "logistic_regression": pipeline.named_steps["logistic_regression"].get_params(),
        "score_interpretation": "uncalibrated model scores", "metadata": metadata,
    })


def load_embedding_matcher(directory):
    names = json.loads((directory / "feature_names.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "matcher_config.json").read_text(encoding="utf-8"))
    pc.require(tuple(names) == ef.FEATURE_NAMES and config["threshold"] == 0.5, "Saved experiment schema/threshold mismatch")
    pipeline = joblib.load(directory / "matcher_pipeline.joblib")
    pc.require(tuple(pipeline.feature_names_in_) == ef.FEATURE_NAMES, "Saved experimental pipeline schema mismatch")
    return pipeline, config["threshold"]


def select_next_model(stage3_metrics, embedding_metrics):
    before, after = stage3_metrics["f1"], embedding_metrics["f1"]
    if after is not None and (before is None or after > before):
        return "embedding_experiment", "higher_validation_f1"
    return "stage3_logistic_regression", ("validation_f1_tie_keep_simpler" if before == after
                                           else "stage3_has_higher_or_defined_validation_f1")


def experiment_versions():
    packages = ("sentence-transformers", "torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                "scikit-learn", "numpy", "pandas", "joblib", "rapidfuzz")
    return {"python": sys.version.split()[0], **{name: version(name) for name in packages}}


def run_real_integration(root=PROJECT_ROOT):
    encoder, context, preparation = ef.prepare_real_encoder(root)
    cache = ef.EmbeddingCache(root / ".cache/embeddings-integration", context)
    texts = ["acme manufacturing ltd", "10 high street", "birch services ltd", "20 low road", ""]
    vectors, first = cache.get_or_encode(texts, ef.encode_function(encoder))
    def no_encoding(texts, batch_size):
        raise AssertionError("Warm real-model cache should not encode again")
    restored, second = cache.get_or_encode(texts, no_encoding)
    pc.require(len(vectors) == 4 and all(vector.shape == (384,) for vector in vectors.values()),
               "Real MiniLM integration dimension/unique-text failure")
    pc.require(all(np.array_equal(vectors[text], restored[text]) for text in vectors), "Real cache round-trip changed vectors")
    report = {"status": "passed", "real_pretrained_model": True, "model_id": ef.MODEL_ID,
              "revision": context.model_revision, "dimension": 384, "device": "cpu",
              "model_preparation": preparation, "first_pass": first, "warm_cache_check": second}
    pc.write_json(root / "reports/embedding_experiment/real_model_integration.json", report)
    print(f"PASS: real CPU MiniLM integration, revision {context.model_revision}, 384 dimensions, cache round-trip")
    return report


def run_experiment(root=PROJECT_ROOT):
    assert_preserved_inputs(root)
    inputs = tm.load_stage3_inputs(root)
    pairs, records, split_ids = inputs["pairs"], inputs["records"], inputs["split_ids"]
    started = perf_counter()
    original_records = {record_id: {name: row[name] for name in pf.NORMALIZED_FIELDS}
                        for record_id, row in records.items()}
    base = {split: pf.build_feature_frame(pairs[split], original_records) for split in tm.EVALUATED_SPLITS}
    base_feature_seconds = perf_counter() - started
    truth = tm.load_selected_truth(root / "data/ground_truth/record_entity_map.csv", set(records))
    pc.require(all(truth[row["record_id"]] == row["true_entity_id"] for row in inputs["assignments"]),
               "Selected saved truth mapping changed")
    labels = {split: tm.build_targets(pairs[split], truth) for split in tm.EVALUATED_SPLITS}
    # Original model is loaded, never fitted. No Stage 3 output path is written.
    baseline, baseline_threshold = tm.load_matcher(root / "models")
    encoder, context, preparation = ef.prepare_real_encoder(root)
    active_ids = {row[key] for subset in pairs.values() for row in subset for key in ("record_id_left", "record_id_right")}
    text_records = ef.project_text_records({record_id: records[record_id] for record_id in active_ids})
    vectors, embedding_timing = ef.EmbeddingCache(root / ".cache/embeddings", context).get_or_encode(
        ef.unique_texts(text_records), ef.encode_function(encoder)
    )
    started = perf_counter()
    extended = {split: ef.extend_pair_features(base[split], pairs[split], text_records, vectors)
                for split in tm.EVALUATED_SPLITS}
    cosine_feature_seconds = perf_counter() - started
    started = perf_counter()
    pipeline = fit_embedding_matcher(extended["train"], labels["train"])
    fit_seconds = perf_counter() - started
    started = perf_counter()
    old_scores, old_predictions = tm.predict_logistic(baseline, base["validation"], baseline_threshold)
    baseline_predict_seconds = perf_counter() - started
    started = perf_counter()
    scores, predictions = predict_embedding(pipeline, extended["validation"])
    embedding_predict_seconds = perf_counter() - started
    saved_baseline = pc.read_csv(root / "reports/matching/validation_predictions.csv", tm.PREDICTION_FIELDS)
    pc.require([(row["record_id_left"], row["record_id_right"]) for row in saved_baseline] ==
               [(row["record_id_left"], row["record_id_right"]) for row in pairs["validation"]],
               "Validation candidate cohort/order differs from Stage 3")
    pc.require(np.array_equal(old_predictions, [int(row["logistic_prediction"]) for row in saved_baseline])
               and np.allclose(old_scores, [float(row["logistic_score"]) for row in saved_baseline], atol=1e-12, rtol=0),
               "Unchanged Stage 3 model no longer reproduces its saved validation predictions")
    total_true, blocking_misses = tm.all_true_pairs_and_blocking_misses(
        {record_id: truth[record_id] for record_id in split_ids["validation"]}, pairs["validation"]
    )
    comparison = {
        "stage3_logistic_regression": tm.evaluate_predictions(labels["validation"], old_predictions, total_true),
        "embedding_experiment": tm.evaluate_predictions(labels["validation"], predictions, total_true),
    }
    chosen, reason = select_next_model(comparison["stage3_logistic_regression"], comparison["embedding_experiment"])
    versions = experiment_versions()
    model_dir, report_dir = root / "models/embedding_experiment", root / "reports/embedding_experiment"
    save_embedding_matcher(pipeline, model_dir, context, {
        "training_candidates": len(base["train"]), "training_class_counts": {
            "match": int(labels["train"].sum()), "non_match": int((labels["train"] == 0).sum())},
        "batch_size": ef.BATCH_SIZE, "cpu_threads": ef.CPU_THREADS,
        "fit_scope": "training only; pretrained encoder frozen; no searches", "versions": versions,
    })
    loaded, loaded_threshold = load_embedding_matcher(model_dir)
    reloaded_scores, reloaded_predictions = predict_embedding(loaded, extended["validation"], loaded_threshold)
    pc.require(np.array_equal(scores, reloaded_scores) and np.array_equal(predictions, reloaded_predictions),
               "Experimental save/load predictions differ")
    pc.write_json(model_dir / "package_versions.json", versions)
    rows = []
    for index, pair in enumerate(pairs["validation"]):
        rows.append({**tm.observed_pair_row(pair, records), "blocking_rules": pair["blocking_rules"],
                     **extended["validation"].iloc[index].to_dict(), "target": int(labels["validation"][index]),
                     "stage3_score": float(old_scores[index]), "stage3_prediction": int(old_predictions[index]),
                     "embedding_score": float(scores[index]), "embedding_prediction": int(predictions[index]),
                     "prediction_changed": int(old_predictions[index] != predictions[index]),
                     "score_delta": float(scores[index] - old_scores[index])})
    changed = [row for row in rows if row["prediction_changed"]]
    pc.write_csv(report_dir / "validation_predictions.csv", PREDICTION_FIELDS, rows)
    pc.write_csv(report_dir / "changed_predictions.csv", PREDICTION_FIELDS, changed)
    report = {
        "data_classification": "SYNTHETIC", "stage": 4, "model_id": ef.MODEL_ID,
        "resolved_revision": context.model_revision, "cache_context": asdict(context),
        "versions": versions, "validation_candidates": len(rows), "all_true_duplicate_pairs": total_true,
        "blocking_misses": len(blocking_misses), "threshold": 0.5,
        "comparison": comparison, "changed_prediction_count": len(changed),
        "maximum_absolute_score_change": float(np.max(np.abs(scores - old_scores))) if len(rows) else None,
        "timings": {"model_preparation": preparation, "embeddings": embedding_timing,
                    "base_feature_seconds": base_feature_seconds, "cosine_feature_seconds": cosine_feature_seconds,
                    "experimental_fit_seconds": fit_seconds,
                    "stage3_validation_prediction_seconds": baseline_predict_seconds,
                    "embedding_validation_prediction_seconds": embedding_predict_seconds},
        "timing_interpretation": "Single-run CPU wall times. Prediction excludes feature building, encoder load, "
                                 "download and fitting. Embedding encode excludes model download/load and cache I/O. "
                                 "Weights may be warm after the separate real integration check. Not a benchmark.",
        "selection": {"chosen_model": chosen, "reason": reason},
        "score_interpretation": "uncalibrated model scores; no claim of embedding improvement without measured gain",
        "test_performance": "Hidden; test candidates/labels never evaluated",
        "checks": {"stage3_predictions_reproduced": True, "save_load_consistency": True,
                   "training_only_fit": True, "protected_artifacts_unchanged": True},
    }
    pc.write_json(report_dir / "comparison.json", report)
    embedding_chosen = chosen == "embedding_experiment"
    pc.write_json(root / "models/next_stage_choice.json", {
        "chosen_model": chosen, "reason": reason, "selection_metric": "validation_f1",
        "model_path": "models/embedding_experiment/matcher_pipeline.joblib" if embedding_chosen else "models/matcher_pipeline.joblib",
        "feature_schema": list(ef.FEATURE_NAMES if embedding_chosen else pf.FEATURE_NAMES),
        "threshold": 0.5, "pretrained_encoder": asdict(context) if embedding_chosen else None,
        "validation_f1": {name: metrics["f1"] for name, metrics in comparison.items()}, "test_used": False,
    })
    assert_preserved_inputs(root)
    print("SYNTHETIC DATA ONLY - controlled frozen CPU embedding experiment")
    print(f"Model revision: {context.model_revision}; 23 features; fixed threshold 0.5")
    for name, metrics in comparison.items():
        print(f"{name}: precision={metrics['precision']}, recall={metrics['recall']}, F1={metrics['f1']}, "
              f"TP/FP/TN/FN={metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}, "
              f"end-to-end recall={metrics['end_to_end_direct_pair_recall']}")
    print(f"Embedding cache: {embedding_timing}; prediction seconds: Stage3={baseline_predict_seconds:.6f}, experiment={embedding_predict_seconds:.6f}")
    print(f"Changed validation predictions: {len(changed)}; selected: {chosen} ({reason})")
    print("PASS: Stage 1/2/3 artifacts preserved; test performance hidden; save/load predictions identical")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-check", action="store_true", help="Run a separate real-model check, not model training")
    arguments = parser.parse_args()
    try:
        run_real_integration() if arguments.integration_check else run_experiment()
    except Exception as error:
        pc.write_json(PROJECT_ROOT / "reports/embedding_experiment/failure.json", {
            "status": "failed", "phase": "real_integration" if arguments.integration_check else "experiment",
            "error_type": type(error).__name__, "error": str(error),
            "model_id": ef.MODEL_ID, "invented_embeddings_used": False,
        })
        print(f"Stage 4 failed: {type(error).__name__}: {error}. No invented embeddings substituted.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
