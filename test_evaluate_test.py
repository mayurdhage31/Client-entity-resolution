"""Stage 6 evaluator checks use temporary artificial data, never real test labels."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

import build_clusters as bc
import evaluate_test as et
import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm


def write_fixture(root):
    """Four true duplicate pairs, one absent candidate, one incorrect graph edge."""
    specifications = [("train1", "T", "train"), ("valid1", "V", "validation"),
                      ("0001", "A", "test"), ("0002", "A", "test"), ("0003", "A", "test"),
                      ("0004", "B", "test"), ("0005", "B", "test"), ("0006", "C", "test")]
    observed, assignments, truth = [], [], []
    for record_id, entity, split in specifications:
        observed.append({"record_id": record_id, "source_system": "SYNTHETIC_FIXTURE",
                         "company_name": f"Fictional {entity} Limited", "registration_id": "00" + entity,
                         "address": f"{entity} High Street", "city": "Fixture City", "postcode": "00123", "country": "UK"})
        assignments.append({"record_id": record_id, "true_entity_id": entity, "split_group_id": f"family-{entity}",
                            "split": split, **pc.split_metadata(42, pc.SPLIT_RATIOS)})
        truth.append({"record_id": record_id, "true_entity_id": entity})
    pc.write_csv(root / "data/raw/client_records.csv", pc.OBSERVED_FIELDS, observed)
    pc.write_csv(root / "data/processed/cleaned_records.csv", pc.CLEANED_FIELDS, pc.clean_records(observed))
    pc.write_csv(root / "data/splits/record_splits.csv", pc.SPLIT_FIELDS, assignments)
    pc.write_csv(root / "data/ground_truth/record_entity_map.csv", pc.TRUTH_FIELDS, truth)
    pairs = [{"record_id_left": left, "record_id_right": right, "blocking_rules": "fuzzy_name_top_k"}
             for left, right in (("0001", "0002"), ("0001", "0004"), ("0002", "0003"), ("0004", "0005"))]
    pc.write_csv(root / "data/processed/candidate_pairs_test.csv", pc.PAIR_FIELDS, pairs)
    for split in ("train", "validation"):
        pc.write_csv(root / f"data/processed/candidate_pairs_{split}.csv", pc.PAIR_FIELDS, [])
    blocking = {"blocking_configuration": pc.asdict(pc.DEFAULT_BLOCKING_CONFIG),
                "fallback_configuration": pc.asdict(pc.DEFAULT_FALLBACK_CONFIG)}
    pc.write_json(root / "reports/blocking_report.json", blocking)
    pc.write_json(root / "reports/older_report.json", {"keep": "unchanged"})
    model = root / "models/matcher_pipeline.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"Artificial placeholder; fixture mocks loading, never a real fitted model")
    choice = {"chosen_model": "stage3_logistic_regression", "model_path": "models/matcher_pipeline.joblib",
              "feature_schema": list(pf.FEATURE_NAMES), "threshold": .5, "test_used": False}
    pc.write_json(root / "models/next_stage_choice.json", choice)
    pc.write_json(root / "models/feature_names.json", list(pf.FEATURE_NAMES))
    pc.write_json(root / "models/matcher_config.json", {
        "prediction_threshold": .5, "feature_version": pf.FEATURE_VERSION,
        "rule_based_baseline": {"description": "Illustrative fixture rules"},
        "training_provenance": {
            "input_sha256": {"data/processed/candidate_pairs_train.csv": et.sha256(root / "data/processed/candidate_pairs_train.csv")},
            "existing_blocking_configuration": blocking["blocking_configuration"],
            "existing_fallback_configuration": blocking["fallback_configuration"],
        },
    })
    (root / "evaluate_test.py").write_text("# Artificial pipeline source for hashing\n", encoding="utf-8")
    return choice


def fixture_predictions(root):
    records, pairs = et.load_test_inputs(root)
    scores, predictions = np.array([.9, .9, .1, .1]), np.array([1, 1, 0, 0])
    features = pf.build_feature_frame(pairs, {key: {field: row[field] for field in pf.NORMALIZED_FIELDS}
                                             for key, row in records.items()})
    graph = bc.build_duplicate_graph(records, pairs, scores, .5)
    _, clusters = bc.cluster_memberships(graph)
    flags = bc.registration_review_flags(clusters, records)
    return records, pairs, scores, predictions, pf.predict_rule_based(features), graph, flags


def freeze_fixture(root, choice, unit_gate=False):
    with patch.object(bc, "load_selected_model", return_value=(object(), .5, choice)), redirect_stdout(io.StringIO()):
        et.create_freeze(root)
    _, digest = et.verify_freeze(root)
    if unit_gate:
        directory = et.version_directory(root, "v1")
        log = directory / "unit_test_log.txt"
        log.write_text("Artificial successful unit gate for temporary evaluator fixture only\n", encoding="utf-8")
        et.exclusive_json(directory / "unit_test_results.json", {
            "freeze_sha256": digest, "success": True, "tests_run": 1, "failures": 0, "errors": 0,
            "skipped": 0, "unit_log_sha256": et.sha256(log),
        })
    return digest


class EvaluatorMetricChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_fixture(self.root)
        self.records, self.pairs, self.scores, self.predictions, self.rules, self.graph, self.flags = fixture_predictions(self.root)
        self.truth = et.load_test_truth(self.root, set(self.records))

    def metrics(self):
        return et.evaluate_fixed_predictions(self.records, self.pairs, self.truth, self.rules,
                                             self.predictions, self.graph, self.flags)[0]

    def test_blocking_denominator_is_all_within_split_pairs(self):
        row = self.metrics()["blocking"]
        self.assertEqual((row["record_count"], row["entity_count"], row["total_possible_pairs"]), (6, 3, 15))
        self.assertEqual((row["candidate_pairs"], row["all_true_duplicate_pairs"], row["true_duplicate_pairs_retrieved"]), (4, 4, 3))
        self.assertEqual(row["blocking_recall"], .75)
        self.assertEqual(row["pair_reduction"], 1 - 4 / 15)
        self.assertEqual((row["candidate_match_count"], row["candidate_non_match_count"]), (3, 1))

    def test_candidate_recall_and_direct_recall_use_different_denominators(self):
        metrics = self.metrics()
        row = metrics["candidate_classification"]["logistic_regression"]
        self.assertEqual([row[key] for key in ("tp", "fp", "tn", "fn")], [1, 1, 0, 2])
        self.assertEqual((row["precision"], row["recall"], row["f1"]), (.5, 1 / 3, .4))
        self.assertEqual(metrics["end_to_end_direct_pair"]["logistic_regression"]["end_to_end_direct_pair_recall"], .25)

    def test_blocking_misses_and_classifier_misses_remain_separate(self):
        metrics = self.metrics()
        direct = metrics["end_to_end_direct_pair"]["logistic_regression"]
        self.assertEqual((direct["blocking_misses"], direct["classifier_misses_on_candidates"], direct["end_to_end_missed_pairs"]), (1, 2, 3))
        self.assertEqual(metrics["candidate_classification"]["rule_based"]["recall"], 1)
        self.assertEqual(metrics["end_to_end_direct_pair"]["rule_based"]["end_to_end_direct_pair_recall"], .75)
        self.assertEqual(len(self.pairs), 4)  # No missed pair was injected.

    def test_singletons_and_one_incorrect_bridge_affect_all_pair_cluster_counts(self):
        row = self.metrics()["clustering"]
        self.assertEqual((row["cluster_count"], row["singleton_count"], row["largest_cluster_size"]), (4, 3, 3))
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (1, 2, 3, 9))
        self.assertEqual((row["pairwise_precision"], row["pairwise_recall"], row["pairwise_f1"]), (1 / 3, .25, 2 / 7))
        self.assertEqual((row["overmerged_cluster_count"], row["fragmented_true_entity_count"], row["review_flag_count"]), (1, 2, 1))

    def test_all_pair_counts_agree_with_explicit_enumeration(self):
        membership, _ = bc.cluster_memberships(self.graph)
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for left, right in combinations(sorted(self.records), 2):
            actual, predicted = self.truth[left] == self.truth[right], membership[left] == membership[right]
            key = "tp" if actual and predicted else "fp" if predicted else "fn" if actual else "tn"
            counts[key] += 1
        row = self.metrics()["clustering"]
        self.assertEqual(counts, {key: row[key] for key in counts})

    def test_zero_denominators_with_singleton_and_no_candidates(self):
        records = {"0001": self.records["0001"]}
        graph = bc.build_duplicate_graph(records, [], [], .5)
        metrics, targets, misses = et.evaluate_fixed_predictions(records, [], {"0001": "A"}, [], [], graph, [])
        self.assertEqual((len(targets), len(misses), metrics["clustering"]["singleton_count"]), (0, 0, 1))
        self.assertIsNone(metrics["blocking"]["pair_reduction"])
        self.assertIsNone(metrics["blocking"]["blocking_recall"])
        self.assertIsNone(metrics["candidate_classification"]["logistic_regression"]["f1"])
        self.assertIsNone(metrics["end_to_end_direct_pair"]["logistic_regression"]["end_to_end_direct_pair_recall"])
        self.assertIsNone(metrics["clustering"]["pairwise_recall"])

    def test_reject_graph_not_derived_from_fixed_predictions(self):
        self.graph.add_edge("0004", "0005")
        with self.assertRaisesRegex(ValueError, "Graph disagrees"):
            self.metrics()


class FrozenArtifactChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.choice = write_fixture(self.root)
        self.digest = freeze_fixture(self.root, self.choice)

    def test_freeze_is_idempotent_and_records_all_required_settings(self):
        directory = et.version_directory(self.root, "v1")
        original = (directory / "freeze_manifest.json").read_bytes()
        with redirect_stdout(io.StringIO()):
            manifest = et.create_freeze(self.root)
        self.assertEqual(original, (directory / "freeze_manifest.json").read_bytes())
        self.assertEqual(manifest["selected_model_sha256"], et.sha256(self.root / self.choice["model_path"]))
        self.assertEqual(manifest["feature_schema"], list(pf.FEATURE_NAMES))
        self.assertEqual(manifest["threshold"], .5)
        self.assertEqual(manifest["clustering"], et.CLUSTERING_POLICY)
        self.assertEqual(manifest["review_flags"], et.REVIEW_POLICY)
        self.assertIn("data/processed/candidate_pairs_test.csv", manifest["protected_sha256"])
        self.assertIn("data/ground_truth/record_entity_map.csv", manifest["protected_sha256"])
        self.assertIn("packages", manifest["runtime_versions"])

    def test_changed_model_candidate_split_source_or_old_report_is_rejected(self):
        for relative in (self.choice["model_path"], "data/processed/candidate_pairs_test.csv",
                         "data/splits/record_splits.csv", "evaluate_test.py", "reports/older_report.json"):
            with self.subTest(relative=relative):
                path = self.root / relative
                before = path.read_bytes()
                path.write_bytes(before + b"changed")
                with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                    et.verify_freeze(self.root)
                path.write_bytes(before)

    def test_added_protected_source_or_deleted_artifact_is_rejected(self):
        extra = self.root / "new_source.py"
        extra.write_text("# Added after freeze", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            et.verify_freeze(self.root)
        extra.unlink()
        (self.root / "models/feature_names.json").unlink()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            et.verify_freeze(self.root)

    def test_manifest_checksum_and_package_changes_are_rejected(self):
        with patch.object(et, "runtime_versions", return_value={"python": "different"}):
            with self.assertRaisesRegex(ValueError, "versions changed"):
                et.verify_freeze(self.root)
        path = et.version_directory(self.root, "v1") / "freeze_manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest checksum changed"):
            et.verify_freeze(self.root)

    def test_new_results_and_readme_do_not_change_the_executable_freeze(self):
        (self.root / "README.md").write_text("Document actual outcomes later", encoding="utf-8")
        pc.write_json(self.root / "reports/test_evaluation/v1/new_result.json", {"new": True})
        pc.write_json(self.root / "outputs/test_evaluation/v1/new_output.json", {"new": True})
        et.verify_freeze(self.root)

    def test_unit_gate_required_and_bound_to_exact_manifest_and_log(self):
        with self.assertRaisesRegex(ValueError, "unit checks required"):
            et.verify_unit_gate(self.root, "v1", self.digest)
        freeze_fixture(self.root, self.choice, unit_gate=True)
        et.verify_unit_gate(self.root, "v1", self.digest)
        with self.assertRaisesRegex(ValueError, "different freeze"):
            et.verify_unit_gate(self.root, "v1", "different")
        (et.version_directory(self.root, "v1") / "unit_test_log.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "log changed"):
            et.verify_unit_gate(self.root, "v1", self.digest)

    def test_failed_gate_cannot_access_real_evaluation_path(self):
        freeze_fixture(self.root, self.choice, unit_gate=True)
        path = et.version_directory(self.root, "v1") / "unit_test_results.json"
        proof = json.loads(path.read_text(encoding="utf-8"))
        proof.update(success=False, failures=1)
        pc.write_json(path, proof)
        with patch.object(et, "load_test_truth", side_effect=AssertionError("Labels must remain unopened")):
            with self.assertRaisesRegex(ValueError, "gate failed"):
                et.run_evaluation(self.root)

    def test_invalid_version_path_and_exclusive_writes(self):
        with self.assertRaisesRegex(ValueError, "Invalid evaluation version"):
            et.version_directory(self.root, "../other")
        path = self.root / "reports/test_evaluation/v1/exclusive.json"
        et.exclusive_json(path, {"first": True})
        with self.assertRaises(FileExistsError):
            et.exclusive_json(path, {"replacement": True})
        self.assertEqual(json.loads(path.read_text()), {"first": True})


class EvaluatorInputChecks(unittest.TestCase):
    def test_input_loader_uses_only_test_observations_and_preserves_leading_zeros(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root)
            with patch.object(tm, "load_selected_truth", side_effect=AssertionError("No labels during prediction")):
                records, pairs = et.load_test_inputs(root)
            self.assertEqual(set(records), {f"{index:04d}" for index in range(1, 7)})
            self.assertEqual(records["0001"]["postcode_normalized"], "00123")
            self.assertEqual(records["0001"]["registration_id_normalized"], "00a")
            self.assertTrue(all("true_entity_id" not in row and "split_group_id" not in row for row in records.values()))
            self.assertEqual(len(pairs), 4)

    def test_family_crossing_and_candidate_crossing_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root)
            path = root / "data/splits/record_splits.csv"
            assignments = pc.read_csv(path, pc.SPLIT_FIELDS)
            assignments[0]["split_group_id"] = assignments[2]["split_group_id"]
            pc.write_csv(path, pc.SPLIT_FIELDS, assignments)
            with self.assertRaisesRegex(ValueError, "family crosses"):
                et.load_test_inputs(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root)
            pc.write_csv(root / "data/processed/candidate_pairs_test.csv", pc.PAIR_FIELDS,
                         [{"record_id_left": "0001", "record_id_right": "valid1", "blocking_rules": "fuzzy_name_top_k"}])
            with self.assertRaisesRegex(ValueError, "outside its split"):
                et.load_test_inputs(root)


class EvaluationLifecycleChecks(unittest.TestCase):
    def test_blind_outputs_precede_truth_and_identical_reruns_preserve_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            choice = write_fixture(root)
            freeze_fixture(root, choice, unit_gate=True)
            real_truth_loader = et.load_test_truth
            calls = []
            def after_blind_outputs(project, ids):
                suffix = "original" if not calls else "reruns/001"
                calls.append(suffix)
                reports = root / "reports/test_evaluation/v1" / suffix
                outputs = root / "outputs/test_evaluation/v1" / suffix
                commit = json.loads((reports / "prediction_commit.json").read_text(encoding="utf-8"))
                self.assertEqual(commit["label_free_output_sha256"], et.artifact_hashes(outputs))
                blind = pc.read_csv(outputs / "blind_candidate_predictions.csv", et.BLIND_FIELDS)
                self.assertTrue(all("target" not in row for row in blind))
                memberships = pc.read_csv(outputs / "clusters/cluster_membership.csv", bc.MEMBERSHIP_FIELDS)
                self.assertEqual({row["record_id"] for row in memberships}, ids)
                self.assertEqual(len(memberships), 6)  # Includes record 0006 with no candidates.
                return real_truth_loader(project, ids)
            with redirect_stdout(io.StringIO()), \
                    patch.object(bc, "load_selected_model", return_value=(object(), .5, choice)), \
                    patch.object(tm, "predict_logistic", return_value=(np.array([.9, .9, .1, .1]), np.array([1, 1, 0, 0]))), \
                    patch.object(tm, "fit_matcher", side_effect=AssertionError("Never fit")), \
                    patch.object(pc, "generate_revised_candidate_pairs", side_effect=AssertionError("Never regenerate")), \
                    patch.object(et, "load_test_truth", side_effect=after_blind_outputs):
                original = et.run_evaluation(root)
                original_paths = [root / "reports/test_evaluation/v1/original", root / "outputs/test_evaluation/v1/original"]
                snapshots = [et.artifact_hashes(path) for path in original_paths]
                with self.assertRaisesRegex(ValueError, "already exists"):
                    et.run_evaluation(root)
                rerun = et.run_evaluation(root, rerun=True)
            self.assertEqual(calls, ["original", "reruns/001"])
            self.assertEqual(original["metrics"], rerun["metrics"])
            self.assertTrue(rerun["reproducibility"]["all_output_bytes_identical"])
            self.assertEqual(snapshots, [et.artifact_hashes(path) for path in original_paths])
            digest = et.sha256(root / "reports/test_evaluation/v1/freeze_manifest.json")
            et.verify_completed_run(*original_paths, digest)
            (original_paths[1] / "candidate_predictions.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outputs changed"):
                et.verify_completed_run(*original_paths, digest)
            with self.assertRaisesRegex(ValueError, "only --rerun"):
                et.create_freeze(root, "v2")

    def test_rerun_rejects_incomplete_original_before_reading_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            choice = write_fixture(root)
            freeze_fixture(root, choice, unit_gate=True)
            with patch.object(et, "load_test_truth", side_effect=AssertionError("No label read")):
                with self.assertRaisesRegex(ValueError, "Original evaluation is incomplete"):
                    et.run_evaluation(root, rerun=True)


if __name__ == "__main__":
    unittest.main()
