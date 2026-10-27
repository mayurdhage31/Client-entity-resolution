"""Stage 6: versioned, frozen evaluation of the SYNTHETIC held-out test split.

Freeze -> offline unit checks -> blind predictions/components -> truth -> metrics.
There is deliberately no fitting, retrieval, threshold search, or graph tuning path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
import unittest
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import distributions
from itertools import combinations
from pathlib import Path

import numpy as np

import build_clusters as bc
import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VERSION = "v1"
CLUSTERING_POLICY = {
    "method": "NetworkX connected components of undirected direct match edges",
    "node_scope": "every observed record in the test split, including isolated nodes",
    "edge_rule": "selected Stage 3 model score >= saved threshold; existing candidates only",
    "cluster_id": "clu_ + SHA-256 of JSON-encoded sorted member record IDs",
    "transitivity": "shared component membership does not imply a directly predicted edge",
    "truth_or_reference_used_for_membership": False,
}
REVIEW_POLICY = {
    "rule": "multiple distinct nonempty normalized registration IDs within the same nonempty normalized country",
    "missing_values": "ignored; missing values never create agreement or conflict",
    "leading_zeros": "preserved and significant",
    "action": "human review flag only; never change graph or membership",
    "approval_status": "predicted clusters are not approved record merges",
}
BLIND_FIELDS = tuple(field for field in tm.PREDICTION_FIELDS if field != "target")
CLUSTER_ERROR_FIELDS = (
    "record_id_left", "record_id_right", "cluster_id_left", "cluster_id_right",
    "true_entity_id_left", "true_entity_id_right", "error_type", *tm.OBSERVED_PAIR_FIELDS,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    # Hashing labels as opaque bytes does not inspect their values or outcomes.
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def exclusive_json(path, value):
    """Never overwrite a freeze, test gate, or completed evaluation artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def version_directory(root, evaluation_version):
    pc.require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", evaluation_version)),
               "Invalid evaluation version; use a short directory-safe name")
    return root / "reports/test_evaluation" / evaluation_version


def runtime_versions():
    return {"python": sys.version.split()[0], "packages": dict(sorted(
        (dist.metadata["Name"], dist.version) for dist in distributions() if dist.metadata["Name"]
    ))}


def protected_paths(root):
    """Protect all previous artifacts, plus evaluator/check sources; exclude caches.

    Stage 6 outputs are independently committed below. README is documentation
    updated after evaluation, so it is explicitly outside the executable freeze.
    """
    paths = set(root.glob("*.py"))
    paths.update(path for path in (root / ".gitignore", root / "requirements-embeddings.txt") if path.is_file())
    for directory in ("data", "models", "reports", "outputs"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and not relative.startswith(("reports/test_evaluation/", "outputs/test_evaluation/")):
                paths.add(path)
    return sorted(path.relative_to(root).as_posix() for path in paths)


def artifact_hashes(directory):
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def verify_freeze(root, evaluation_version=DEFAULT_VERSION):
    directory = version_directory(root, evaluation_version)
    path = directory / "freeze_manifest.json"
    pc.require(path.is_file(), "Freeze manifest missing; run --freeze first")
    digest = sha256(path)
    pc.require((directory / "freeze_manifest.sha256").read_text(encoding="utf-8").strip() == digest,
               "Immutable freeze manifest checksum changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pc.require(manifest["evaluation_version"] == evaluation_version, "Freeze version mismatch")
    pc.require(protected_paths(root) == sorted(manifest["protected_sha256"]),
               "Frozen artifact inventory changed (added/removed protected file)")
    for relative, expected in manifest["protected_sha256"].items():
        pc.require(sha256(root / relative) == expected, f"Frozen artifact changed: {relative}")
    pc.require(runtime_versions() == manifest["runtime_versions"], "Frozen package/Python versions changed")
    pc.require(manifest["feature_schema"] == list(pf.FEATURE_NAMES)
               and manifest["clustering"] == CLUSTERING_POLICY and manifest["review_flags"] == REVIEW_POLICY,
               "Frozen schema or graph/review policy changed")
    return manifest, digest


def create_freeze(root=PROJECT_ROOT, evaluation_version=DEFAULT_VERSION):
    directory = version_directory(root, evaluation_version)
    if (directory / "freeze_manifest.json").exists():
        manifest, _ = verify_freeze(root, evaluation_version)
        print(f"Existing immutable {evaluation_version} freeze verified; no overwrite")
        return manifest
    pc.require(not any((root / "reports/test_evaluation").glob("*/original/evaluation.json")),
               "Held-out evaluation already completed; only --rerun of its existing freeze is allowed")
    _, threshold, choice = bc.load_selected_model(root)
    configuration = json.loads((root / "models/matcher_config.json").read_text(encoding="utf-8"))
    blocking = json.loads((root / "reports/blocking_report.json").read_text(encoding="utf-8"))
    provenance = configuration["training_provenance"]
    pc.require(blocking["blocking_configuration"] == provenance["existing_blocking_configuration"]
               and blocking["fallback_configuration"] == provenance["existing_fallback_configuration"],
               "Current blocking settings disagree with selected model provenance")
    # Also verify that data/candidates used to train the saved model are unchanged.
    for relative, expected in provenance["input_sha256"].items():
        pc.require(sha256(root / relative) == expected, f"Model training input changed: {relative}")
    hashes = {relative: sha256(root / relative) for relative in protected_paths(root)}
    manifest = {
        "data_classification": "SYNTHETIC", "stage": 6, "evaluation_version": evaluation_version,
        "created_at_utc": utc_now(), "status": "frozen BEFORE held-out label evaluation",
        "selected_model": choice, "selected_model_sha256": hashes[choice["model_path"]],
        "feature_schema": list(pf.FEATURE_NAMES), "feature_version": pf.FEATURE_VERSION,
        "threshold": threshold, "threshold_comparison": ">=",
        "rule_based_baseline": configuration["rule_based_baseline"],
        "blocking": {"exact": blocking["blocking_configuration"], "fallback": blocking["fallback_configuration"],
                     "candidate_policy": "consume saved candidate files; never regenerate or add missed pairs"},
        "clustering": CLUSTERING_POLICY, "review_flags": REVIEW_POLICY,
        "runtime_versions": runtime_versions(), "protected_sha256": hashes,
        "documentation_excluded": "README.md may be updated with actual results; caches and new Stage 6 outputs are excluded",
        "label_policy": "Truth files hashed as opaque bytes only; labels joined after blind predictions/components are saved",
        "rerun_policy": "Only identical frozen pipeline/data/environment; new labelled rerun directories, no overwrites",
    }
    exclusive_json(directory / "freeze_manifest.json", manifest)
    with (directory / "freeze_manifest.sha256").open("x", encoding="utf-8") as handle:
        handle.write(sha256(directory / "freeze_manifest.json") + "\n")
    verify_freeze(root, evaluation_version)
    print(f"Saved immutable {evaluation_version} freeze: {len(hashes)} protected artifacts; no test outcomes read")
    return manifest


def check_unit_suite(root=PROJECT_ROOT, evaluation_version=DEFAULT_VERSION):
    """One complete offline suite, with a persisted gate tied to this exact freeze."""
    _, freeze_digest = verify_freeze(root, evaluation_version)
    directory = version_directory(root, evaluation_version)
    pc.require(not (directory / "unit_test_results.json").exists(), "Unit suite already recorded for this version")
    pc.require(not (directory / "original").exists(), "Unit checks must precede the real test evaluation")
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py", top_level_dir=str(root))
    stream = io.StringIO()
    started, start_time = utc_now(), time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start_time
    print(stream.getvalue(), end="")
    with (directory / "unit_test_log.txt").open("x", encoding="utf-8") as handle:
        handle.write(stream.getvalue())
    proof = {
        "freeze_sha256": freeze_digest, "complete_suite_runs": 1, "started_at_utc": started,
        "completed_at_utc": utc_now(), "runtime_seconds": elapsed,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "success": result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not result.expectedFailures,
        "unit_log_sha256": sha256(directory / "unit_test_log.txt"),
        "scope": "complete unit suite; artificial temporary fixtures only, no real test outcomes",
    }
    exclusive_json(directory / "unit_test_results.json", proof)
    verify_freeze(root, evaluation_version)
    pc.require(proof["success"], "Unit checks failed; do not evaluate real test outcomes. Fix evaluator and freeze a new version")
    return proof


def verify_unit_gate(root, evaluation_version, freeze_digest):
    directory = version_directory(root, evaluation_version)
    path = directory / "unit_test_results.json"
    pc.require(path.is_file(), "Complete unit checks required before test evaluation; run --check")
    proof = json.loads(path.read_text(encoding="utf-8"))
    pc.require(proof["freeze_sha256"] == freeze_digest and proof["success"] and proof["tests_run"] > 0
               and proof["failures"] == proof["errors"] == proof["skipped"] == 0,
               "Unit-check gate failed or belongs to a different freeze")
    pc.require(sha256(directory / "unit_test_log.txt") == proof["unit_log_sha256"], "Unit-check log changed")
    return proof


def load_test_inputs(root):
    """Project split metadata WITHOUT accessing its embedded true_entity_id cells.

    Family IDs guard split isolation only. They are never feature/graph inputs.
    All CSV cells remain text, including postcodes and registration IDs.
    """
    with (root / "data/splits/record_splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        pc.require(len(header) == len(set(header)) and set(header) == set(pc.SPLIT_FIELDS), "Unexpected split schema")
        indexes = {field: header.index(field) for field in ("record_id", "split", "split_group_id")}
        assignments = []
        for cells in reader:
            pc.require(len(cells) == len(header), "Malformed saved split row")
            assignments.append({field: cells[index] for field, index in indexes.items()})
    pc.validate_record_ids(assignments, "Saved splits")
    family_splits = defaultdict(set)
    for row in assignments:
        pc.require(row["split"] in pc.SPLIT_NAMES and bool(row["split_group_id"]), "Invalid saved split metadata")
        family_splits[row["split_group_id"]].add(row["split"])
    pc.require(all(len(values) == 1 for values in family_splits.values()), "Corporate family crosses saved splits")
    test_ids = {row["record_id"] for row in assignments if row["split"] == "test"}
    pc.require(bool(test_ids), "No held-out test records in saved assignments")
    rows = pc.read_csv(root / "data/processed/cleaned_records.csv", pc.CLEANED_FIELDS)
    pc.validate_record_ids(rows, "Cleaned records")
    pc.require({row["record_id"] for row in rows} == {row["record_id"] for row in assignments},
               "Cleaned records and saved splits disagree")
    records = {row["record_id"]: row for row in rows if row["record_id"] in test_ids}
    pairs = pc.read_csv(root / "data/processed/candidate_pairs_test.csv", pc.PAIR_FIELDS)
    pc.validate_candidate_pairs(pairs, test_ids)
    return records, pairs


def load_test_truth(root, allowed_ids):
    # This is called only AFTER label-free predictions and graph outputs are saved.
    truth = tm.load_selected_truth(root / "data/ground_truth/record_entity_map.csv", allowed_ids)
    assignments = pc.read_csv(root / "data/splits/record_splits.csv", pc.SPLIT_FIELDS)
    entity_splits = defaultdict(set)
    for row in assignments:
        pc.require(bool(row["true_entity_id"]), "Saved split entity label missing")
        entity_splits[row["true_entity_id"]].add(row["split"])
        if row["record_id"] in allowed_ids:
            pc.require(row["split"] == "test" and row["true_entity_id"] == truth[row["record_id"]],
                       "Held-out truth and saved assignment disagree")
    pc.require(all(len(values) == 1 for values in entity_splits.values()), "Entity crosses saved splits")
    return truth


def evaluate_fixed_predictions(records, pairs, truth, rule_predictions, model_predictions, graph, flags):
    """Reuse fixed Stage 3/5 metric definitions; test blocking is evaluated here.

    The older Stage 2 entry point deliberately retains its train/validation guard.
    """
    pc.require(set(records) == set(truth) == set(graph), "Test record/truth/graph coverage mismatch")
    pc.validate_candidate_pairs(pairs, set(records))
    targets = tm.build_targets(pairs, truth)
    all_true, blocking_misses = tm.all_true_pairs_and_blocking_misses(truth, pairs)
    possible = math.comb(len(records), 2)
    retrieved = int(targets.sum())
    blocking = {
        "record_count": len(records), "entity_count": len(set(truth.values())),
        "total_possible_pairs": possible, "candidate_pairs": len(pairs),
        "pair_reduction": 1 - len(pairs) / possible if possible else None,
        "all_true_duplicate_pairs": all_true, "true_duplicate_pairs_retrieved": retrieved,
        "missed_true_duplicate_pairs": len(blocking_misses),
        "candidate_match_count": retrieved, "candidate_non_match_count": len(pairs) - retrieved,
        "blocking_recall": retrieved / all_true if all_true else None,
        "candidate_pairs_by_rule": {rule: sum(rule in row["blocking_rules"].split("|") for row in pairs)
                                    for rule in pc.ALL_METHOD_NAMES},
    }
    candidate, direct = {}, {}
    for method, predictions in (("rule_based", rule_predictions), ("logistic_regression", model_predictions)):
        result = tm.evaluate_predictions(targets, predictions, all_true)
        candidate[method] = {key: result[key] for key in (
            "precision", "recall", "f1", "tp", "fp", "tn", "fn", "candidate_class_counts", "predicted_class_counts"
        )}
        candidate[method]["undefined_metrics"] = [key for key in ("precision", "recall", "f1") if result[key] is None]
        direct[method] = {key: result[key] for key in (
            "end_to_end_direct_pair_recall", "all_true_duplicate_pairs", "blocking_misses",
            "classifier_misses_on_candidates", "end_to_end_missed_pairs"
        )}
        direct[method]["correct_direct_predicted_matches"] = result["tp"]
    expected_edges = {tuple(sorted((pair["record_id_left"], pair["record_id_right"])))
                      for pair, predicted in zip(pairs, model_predictions) if predicted == 1}
    pc.require({tuple(sorted(edge)) for edge in graph.edges} == expected_edges, "Graph disagrees with fixed predictions")
    clusters = bc.evaluate_clusters(graph, truth)
    _, membership_groups = bc.cluster_memberships(graph)
    summaries = bc.cluster_summaries(graph, membership_groups, flags)
    clusters.update({
        "direct_predicted_edges": graph.number_of_edges(),
        "inferred_without_direct_edge_pairs": sum(row["inferred_without_direct_edge_count"] for row in summaries),
        "review_flag_count": len(flags), "clusters_requiring_review": len({row["cluster_id"] for row in flags}),
    })
    return {"blocking": blocking, "candidate_classification": candidate,
            "end_to_end_direct_pair": direct, "clustering": clusters}, targets, blocking_misses


def save_errors(directory, records, pairs, features, targets, rules, scores, predictions, truth, membership, misses):
    rows = tm.validation_prediction_rows(pairs, records, features, targets, rules, scores, predictions)
    pc.write_csv(directory / "candidate_predictions.csv", tm.PREDICTION_FIELDS, rows)
    errors = {}
    for method, column in (("rule_based", "rule_based_prediction"), ("logistic_regression", "logistic_prediction")):
        errors[method] = {}
        for name, target, predicted in (("false_positives", 0, 1), ("false_negatives", 1, 0)):
            selected = [row for row in rows if row["target"] == target and row[column] == predicted]
            relative = f"errors/{method}_{name}.csv"
            pc.write_csv(directory / relative, tm.PREDICTION_FIELDS, selected)
            errors[method][name] = {"count": len(selected), "file": relative}
    miss_rows = [{**tm.observed_pair_row(pair, records), "target": 1,
                  "status": "blocking_miss_never_scored_or_injected"} for pair in misses]
    pc.write_csv(directory / "errors/blocking_misses.csv", tm.BLOCKING_MISS_FIELDS, miss_rows)
    errors["blocking_misses"] = {"count": len(miss_rows), "file": "errors/blocking_misses.csv"}
    cluster_errors = {"false_positives": [], "false_negatives": []}
    for left, right in combinations(sorted(records), 2):
        same_cluster, same_entity = membership[left] == membership[right], truth[left] == truth[right]
        if same_cluster != same_entity:
            error = "false_positives" if same_cluster else "false_negatives"
            pair = {"record_id_left": left, "record_id_right": right}
            cluster_errors[error].append({**tm.observed_pair_row(pair, records),
                "cluster_id_left": membership[left], "cluster_id_right": membership[right],
                "true_entity_id_left": truth[left], "true_entity_id_right": truth[right], "error_type": error})
    errors["clustering"] = {}
    for error, selected in cluster_errors.items():
        relative = f"errors/cluster_{error}.csv"
        pc.write_csv(directory / relative, CLUSTER_ERROR_FIELDS, selected)
        errors["clustering"][error] = {"count": len(selected), "file": relative}
    return errors


def readable_report(report):
    metrics = report["metrics"]
    block, clusters = metrics["blocking"], metrics["clustering"]
    def percent(value):
        return "undefined (zero denominator)" if value is None else f"{100 * value:.4f}%"
    lines = ["# Frozen held-out test evaluation — SYNTHETIC DATA", "",
        f"Run: {report['run_kind']}. Version: {report['evaluation_version']}.", "",
        f"Model: {report['selected_model']}; unchanged threshold {report['threshold']}; 21 observed-only features.", "",
        "## Blocking", "",
        f"{block['record_count']} records / {block['entity_count']} entities; {block['total_possible_pairs']} possible unordered pairs; "
        f"{block['candidate_pairs']} existing candidates; pair reduction {percent(block['pair_reduction'])}.", "",
        f"True pairs retrieved {block['true_duplicate_pairs_retrieved']} / {block['all_true_duplicate_pairs']}; "
        f"missed {block['missed_true_duplicate_pairs']}; blocking recall {percent(block['blocking_recall'])}.", "",
        "## Candidate classification and direct-pair recall", "",
        "| Method | Precision | Candidate recall | F1 | TP / FP / TN / FN | End-to-end direct recall | Blocking / classifier misses |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for method, row in metrics["candidate_classification"].items():
        direct = metrics["end_to_end_direct_pair"][method]
        lines.append(f"| {method} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | "
                     f"{row['tp']} / {row['fp']} / {row['tn']} / {row['fn']} | "
                     f"{percent(direct['end_to_end_direct_pair_recall'])} | "
                     f"{direct['blocking_misses']} / {direct['classifier_misses_on_candidates']} |")
    lines += ["", "Candidate recall uses true duplicates among candidates. End-to-end direct recall uses ALL true duplicate "
        "pairs in the split. Blocking misses were never scored or added to candidates. Scores are uncalibrated; rules are illustrative, not banking policy.",
        "", "## Connected-component clusters", "",
        f"{clusters['cluster_count']} clusters, {clusters['singleton_count']} singletons, largest {clusters['largest_cluster_size']}; "
        f"{clusters['direct_predicted_edges']} direct edges; {clusters['inferred_without_direct_edge_pairs']} additional same-component pairs.", "",
        f"All-pairs precision {percent(clusters['pairwise_precision'])}; recall {percent(clusters['pairwise_recall'])}; "
        f"F1 {percent(clusters['pairwise_f1'])}; TP / FP / FN = {clusters['tp']} / {clusters['fp']} / {clusters['fn']}.", "",
        f"Overmerged clusters: {clusters['overmerged_cluster_count']}; fragmented entities: {clusters['fragmented_true_entity_count']}; "
        f"review flags: {clusters['review_flag_count']} across {clusters['clusters_requiring_review']} clusters.", "",
        "Cluster metrics cover ALL unordered record pairs. Transitive co-membership is separate from a directly predicted edge. "
        "Registration conflicts flag human review without changing membership. Predictions are not approved record merges.", "",
        "## Integrity and limitations", "",
        f"Freeze SHA-256: `{report['freeze_sha256']}`. {report['protected_file_count']} protected artifacts verified unchanged. "
        f"Complete unit suite before evaluation: {report['unit_tests_passed']} passes.", "",
        "Truth was joined only after label-free predictions, all test nodes, components and flags were committed. "
        "No retraining, retrieval change, threshold tuning or hierarchy-based merge occurred. Zero denominators are JSON null.", "",
        "This small synthetic split represents one generator and one held-out partition. Results do not measure real-bank performance "
        "or production readiness. Connected components can amplify a single bad edge. Test results must not be used to tune this pipeline.", ""]
    return "\n".join(lines)


def verify_completed_run(report_directory, output_directory, freeze_digest):
    path = report_directory / "artifacts.json"
    pc.require(path.is_file(), "Original evaluation is incomplete; refusing a reproducibility rerun")
    committed = json.loads(path.read_text(encoding="utf-8"))
    pc.require(committed["freeze_sha256"] == freeze_digest, "Original evaluation belongs to a different freeze")
    pc.require(artifact_hashes(output_directory) == committed["output_sha256"], "Original evaluation outputs changed")
    # The commit records all reports except its own hash; no self-referential checksum.
    reports = artifact_hashes(report_directory)
    reports.pop("artifacts.json", None)
    pc.require(reports == committed["report_sha256"], "Original evaluation reports changed")
    return committed


def run_evaluation(root=PROJECT_ROOT, evaluation_version=DEFAULT_VERSION, rerun=False):
    manifest, freeze_digest = verify_freeze(root, evaluation_version)
    unit_proof = verify_unit_gate(root, evaluation_version, freeze_digest)
    version_dir = version_directory(root, evaluation_version)
    output_base = root / "outputs/test_evaluation" / evaluation_version
    original_reports, original_outputs = version_dir / "original", output_base / "original"
    if rerun:
        original_commit = verify_completed_run(original_reports, original_outputs, freeze_digest)
        number = 1
        while (version_dir / f"reruns/{number:03d}").exists() or (output_base / f"reruns/{number:03d}").exists():
            number += 1
        suffix = f"reruns/{number:03d}"
        run_kind = f"identical reproducibility rerun {number:03d}"
    else:
        pc.require(not original_reports.exists() and not original_outputs.exists(),
                   "Original evaluation already exists; use --rerun for an explicitly labelled identical rerun")
        suffix, run_kind = "original", "original held-out evaluation"
    pipeline, threshold, choice = bc.load_selected_model(root)
    pc.require(choice == manifest["selected_model"] and threshold == manifest["threshold"], "Selection differs from freeze")
    records, pairs = load_test_inputs(root)
    feature_records = {record_id: {field: row[field] for field in pf.NORMALIZED_FIELDS}
                       for record_id, row in records.items()}
    features = pf.build_feature_frame(pairs, feature_records)
    rules = pf.predict_rule_based(features)
    scores, predictions = tm.predict_logistic(pipeline, features, threshold)
    graph = bc.build_duplicate_graph(records, pairs, scores, threshold)
    membership, clusters = bc.cluster_memberships(graph)
    flags = bc.registration_review_flags(clusters, records)
    summaries = bc.cluster_summaries(graph, clusters, flags)
    report_directory, output_directory = version_dir / suffix, output_base / suffix
    report_directory.mkdir(parents=True, exist_ok=False)
    output_directory.mkdir(parents=True, exist_ok=False)
    exclusive_json(report_directory / "run_started.json", {
        "run_kind": run_kind, "started_at_utc": utc_now(), "freeze_sha256": freeze_digest,
        "unit_test_results_sha256": sha256(version_dir / "unit_test_results.json"),
    })
    blind_rows = []
    for index, pair in enumerate(pairs):
        blind_rows.append({**tm.observed_pair_row(pair, records), "blocking_rules": pair["blocking_rules"],
                           **features.iloc[index].to_dict(), "rule_based_prediction": int(rules[index]),
                           "logistic_score": float(scores[index]), "logistic_prediction": int(predictions[index])})
    pc.write_csv(output_directory / "blind_candidate_predictions.csv", BLIND_FIELDS, blind_rows)
    bc.save_prediction_outputs(output_directory / "clusters", graph, records, membership, clusters, flags, summaries)
    exclusive_json(report_directory / "prediction_commit.json", {
        "freeze_sha256": freeze_digest, "status": "predictions/components/flags saved BEFORE accessing test entity labels",
        "label_free_output_sha256": artifact_hashes(output_directory),
    })
    verify_freeze(root, evaluation_version)

    # FIRST semantic access to test truth. Everything predicted above is already fixed.
    truth = load_test_truth(root, set(records))
    metrics, targets, misses = evaluate_fixed_predictions(records, pairs, truth, rules, predictions, graph, flags)
    errors = save_errors(output_directory, records, pairs, features, targets, rules, scores, predictions, truth, membership, misses)
    # Independent error-table counts must agree with the reused metric functions.
    for error, metric in (("false_positives", "fp"), ("false_negatives", "fn")):
        pc.require(errors["clustering"][error]["count"] == metrics["clustering"][metric], "Cluster error count disagrees")
        for method in ("rule_based", "logistic_regression"):
            pc.require(errors[method][error]["count"] == metrics["candidate_classification"][method][metric],
                       "Classifier error count disagrees")
    report = {
        "data_classification": "SYNTHETIC", "stage": 6, "evaluation_version": evaluation_version,
        "run_kind": run_kind, "completed_at_utc": utc_now(), "freeze_sha256": freeze_digest,
        "selected_model": choice["chosen_model"], "threshold": threshold, "feature_schema": list(pf.FEATURE_NAMES),
        "metrics": metrics, "protected_file_count": len(manifest["protected_sha256"]),
        "unit_tests_passed": unit_proof["tests_run"], "truth_access": "after blind prediction/cluster artifact commit",
        "undefined_metrics": "null for zero denominators; F1=2TP/(2TP+FP+FN)",
        "no_training_tuning_or_retrieval_changes": True,
    }
    if rerun:
        original = json.loads((original_reports / "evaluation.json").read_text(encoding="utf-8"))
        pc.require(metrics == original["metrics"] and artifact_hashes(output_directory) == original_commit["output_sha256"],
                   "Reproducibility mismatch; original results preserved; rerun is NOT accepted as identical")
        report["reproducibility"] = {"metrics_identical": True, "all_output_bytes_identical": True,
                                     "original_artifacts_sha256": sha256(original_reports / "artifacts.json")}
    exclusive_json(report_directory / "evaluation.json", report)
    exclusive_json(report_directory / "error_analysis.json", errors)
    with (report_directory / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(readable_report(report))
    verify_freeze(root, evaluation_version)
    # Verify the earlier label-free commit too: evaluation must not edit predictions.
    commit = json.loads((report_directory / "prediction_commit.json").read_text(encoding="utf-8"))
    pc.require(all(sha256(output_directory / relative) == digest
                   for relative, digest in commit["label_free_output_sha256"].items()), "Blind predictions changed after truth access")
    exclusive_json(report_directory / "verification.json", {
        "freeze_sha256": freeze_digest, "protected_artifacts_unchanged": len(manifest["protected_sha256"]),
        "blind_predictions_unchanged_after_evaluation": True, "all_test_nodes_included": len(graph) == len(records),
        "no_retraining": True, "no_tuning": True, "no_candidate_regeneration": True,
    })
    exclusive_json(report_directory / "artifacts.json", {
        "freeze_sha256": freeze_digest, "output_sha256": artifact_hashes(output_directory),
        "report_sha256": artifact_hashes(report_directory),
    })
    verify_completed_run(report_directory, output_directory, freeze_digest)
    if rerun:
        verify_completed_run(original_reports, original_outputs, freeze_digest)
    print(readable_report(report))
    print(f"Saved {output_directory.relative_to(root)} and {report_directory.relative_to(root)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=DEFAULT_VERSION)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--freeze", action="store_true", help="write/verify immutable manifest without evaluating labels")
    actions.add_argument("--check", action="store_true", help="run complete unit suite ONCE and save the pre-evaluation gate")
    actions.add_argument("--rerun", action="store_true", help="explicit identical reproducibility rerun; preserve original")
    args = parser.parse_args()
    try:
        if args.freeze:
            create_freeze(PROJECT_ROOT, args.evaluation_version)
        elif args.check:
            check_unit_suite(PROJECT_ROOT, args.evaluation_version)
        else:
            run_evaluation(PROJECT_ROOT, args.evaluation_version, args.rerun)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(1, f"Frozen evaluation stopped: {error}\n")


if __name__ == "__main__":
    main()
