"""Focused Stage 3 tests; all model fitting here uses small synthetic fixtures."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm


def normalized(**values):
    return {f"{field}_normalized": values.get(field, "") for field in pf.COMPARED_FIELDS}


def training_features():
    a = normalized(company_name="acme manufacturing ltd", registration_id="001", country="gb",
                   address="10 high street", city="london", postcode="00123")
    b = normalized(company_name="acme manufacturing ltd", country="gb", address="10 high street")
    c = normalized(company_name="birch services ltd", registration_id="002", country="gb",
                   address="99 low road", city="other city", postcode="00456")
    return pd.DataFrame([pf.build_pair_features(left, right) for left, right in ((a, a), (a, b), (a, c), (b, c))],
                        columns=list(pf.FEATURE_NAMES)), np.array([1, 1, 0, 0])


class FeatureChecks(unittest.TestCase):
    def test_symmetry_for_every_feature_and_pair_order(self):
        records = {
            "opaque-a": normalized(company_name="alpha beta ltd", registration_id="001", country="gb",
                                    address="10 high street", city="london"),
            "opaque-b": normalized(company_name="beta alpha ltd", registration_id="002", country="gb",
                                    address="high street 10", postcode="00123"),
        }
        self.assertEqual(pf.build_pair_features(*records.values()), pf.build_pair_features(*reversed(list(records.values()))))
        forward = pf.build_feature_frame([{"record_id_left": "opaque-a", "record_id_right": "opaque-b"}], records)
        reverse = pf.build_feature_frame([{"record_id_left": "opaque-b", "record_id_right": "opaque-a"}], records)
        pd.testing.assert_frame_equal(forward, reverse)

    def test_similarity_scaling_and_token_order(self):
        features = pf.build_pair_features(normalized(company_name="alpha beta", address="high street"),
                                           normalized(company_name="beta alpha", address="street high"))
        self.assertEqual(features["company_name_token_sort_ratio"], 1)
        self.assertEqual(features["address_token_sort_ratio"], 1)
        self.assertLess(features["company_name_ratio"], 1)
        self.assertTrue(all(0 <= value <= 1 for value in features.values()))
        self.assertEqual(tuple(features), pf.FEATURE_NAMES)
        self.assertEqual(len(features), 21)

    def test_both_missing_never_agrees_or_has_high_similarity(self):
        features = pf.build_pair_features(normalized(), normalized())
        for field in pf.COMPARED_FIELDS:
            self.assertEqual(features[f"{field}_both_missing"], 1)
            self.assertEqual(features[f"{field}_one_missing"], 0)
        for name in pf.FEATURE_NAMES:
            if not name.endswith("both_missing"):
                self.assertEqual(features[name], 0)

    def test_one_missing_none_and_whitespace_are_explicit(self):
        filled = normalized(**{field: "present" for field in pf.COMPARED_FIELDS})
        missing = normalized(**{field: None if index % 2 else " \t" for index, field in enumerate(pf.COMPARED_FIELDS)})
        features = pf.build_pair_features(filled, missing)
        for field in pf.COMPARED_FIELDS:
            self.assertEqual(features[f"{field}_one_missing"], 1)
            self.assertEqual(features[f"{field}_both_missing"], 0)
        self.assertEqual(features["registration_id_disagreement"], 0)
        self.assertEqual(features["company_name_ratio"], 0)
        self.assertEqual(features["address_ratio"], 0)

    def test_leading_zero_identifiers_remain_distinct(self):
        features = pf.build_pair_features(normalized(registration_id="0012", postcode="00123"),
                                           normalized(registration_id="12", postcode="123"))
        self.assertEqual(features["registration_id_exact"], 0)
        self.assertEqual(features["registration_id_disagreement"], 1)
        self.assertEqual(features["postcode_exact"], 0)
        with self.assertRaises(TypeError):
            pf.build_pair_features(normalized(registration_id=12), normalized(registration_id="12"))

    def test_record_feature_boundary_rejects_metadata(self):
        for forbidden in ("record_id", "true_entity_id", "split_group_id", "split", "parent_entity_id",
                          "canonical_company_name", "source_system", "blocking_rules"):
            with self.subTest(field=forbidden), self.assertRaisesRegex(ValueError, "only the six"):
                pf.build_pair_features({**normalized(), forbidden: "SECRET"}, normalized())

    def test_matrix_uses_only_allowlist_ignoring_pair_metadata(self):
        records = {"0001": normalized(company_name="acme"), "0002": normalized(company_name="acme")}
        pair = {"record_id_left": "0001", "record_id_right": "0002", "blocking_rules": "ignored"}
        a = pf.build_feature_frame([pair], records)
        b = pf.build_feature_frame([{**pair, "target": 1, "split": "test", "true_entity_id": "SECRET"}], records)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(tuple(a.columns), pf.FEATURE_NAMES)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in a.dtypes))

    def test_model_matrix_rejects_extra_reordered_non_numeric_and_nan_features(self):
        features, _ = training_features()
        invalid = [features.assign(true_entity_id="LABEL"), features[list(reversed(features.columns))],
                   features.assign(company_name_ratio="text"), features.assign(company_name_ratio=np.nan)]
        for frame in invalid:
            with self.subTest(columns=list(frame.columns)), self.assertRaises(ValueError):
                pf.validate_feature_frame(frame)


class RuleChecks(unittest.TestCase):
    def test_registration_rule_and_country_requirement(self):
        rows = [pf.build_pair_features(normalized(registration_id="001", country=country),
                                      normalized(registration_id="001", country=country))
                for country in ("gb", "")]
        frame = pd.DataFrame(rows, columns=list(pf.FEATURE_NAMES))
        np.testing.assert_array_equal(pf.predict_rule_based(frame), [1, 0])

    def test_similarity_rule_boundaries_and_registration_disagreement(self):
        base = pf.build_pair_features(normalized(company_name="acme", address="10 street", country="gb"),
                                     normalized(company_name="acme", address="10 street", country="gb"))
        rows = [dict(base, company_name_ratio=0.95, address_ratio=0.90),
                dict(base, company_name_ratio=0.94999), dict(base, address_ratio=0.89999),
                dict(base, registration_id_disagreement=1), dict(base, country_exact=0)]
        np.testing.assert_array_equal(pf.predict_rule_based(pd.DataFrame(rows, columns=list(pf.FEATURE_NAMES))),
                                      [1, 0, 0, 0, 0])


class ModelChecks(unittest.TestCase):
    def test_scaler_and_model_fit_training_only(self):
        features, targets = training_features()
        model = tm.fit_matcher(features, targets)
        scaler = model.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, features.mean().to_numpy())
        self.assertEqual(int(scaler.n_samples_seen_), len(features))
        validation = pd.DataFrame([pf.build_pair_features(normalized(), normalized())], columns=list(pf.FEATURE_NAMES))
        before = scaler.mean_.copy()
        tm.predict_logistic(model, validation)
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertEqual(int(scaler.n_samples_seen_), len(features))
        self.assertEqual(model.named_steps["logistic_regression"].random_state, 42)
        self.assertEqual(model.named_steps["logistic_regression"].max_iter, 1000)

    def test_threshold_is_fixed_inclusive_and_extra_features_are_rejected(self):
        features, _ = training_features()
        model = SimpleNamespace(classes_=np.array([0, 1]),
                                predict_proba=lambda frame: np.array([[0.5, 0.5], [0.5001, 0.4999]]))
        scores, labels = tm.predict_logistic(model, features.iloc[:2])
        np.testing.assert_array_equal(labels, [1, 0])
        self.assertEqual(scores[0], 0.5)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            tm.predict_logistic(model, features.assign(record_id="SECRET"))

    def test_save_load_prediction_consistency_and_saved_metadata(self):
        features, targets = training_features()
        model = tm.fit_matcher(features, targets)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            configuration = tm.save_matcher(model, directory, {"fixture": "SYNTHETIC"})
            restored, threshold = tm.load_matcher(directory)
            scores, labels = tm.predict_logistic(model, features)
            restored_scores, restored_labels = tm.predict_logistic(restored, features, threshold)
            np.testing.assert_array_equal(scores, restored_scores)
            np.testing.assert_array_equal(labels, restored_labels)
            self.assertEqual(threshold, 0.5)
            self.assertEqual(configuration["pipeline_order"], ["StandardScaler", "LogisticRegression"])
            self.assertEqual(json.loads((directory / "feature_names.json").read_text()), list(pf.FEATURE_NAMES))
            self.assertIn("scikit-learn", json.loads((directory / "package_versions.json").read_text()))
            (directory / "feature_names.json").write_text(json.dumps(list(reversed(pf.FEATURE_NAMES))))
            with self.assertRaisesRegex(ValueError, "allowlist/order"):
                tm.load_matcher(directory)

    def test_fit_rejects_single_class_and_extra_columns(self):
        features, _ = training_features()
        with self.assertRaisesRegex(ValueError, "both match and non-match"):
            tm.fit_matcher(features, np.ones(len(features), dtype=int))
        with self.assertRaisesRegex(ValueError, "allowlist"):
            tm.fit_matcher(features.assign(target=1), [1, 1, 0, 0])


class EvaluationChecks(unittest.TestCase):
    def test_candidate_metrics_and_blocking_vs_classifier_misses(self):
        metrics = tm.evaluate_predictions([1, 1, 0, 0], [1, 0, 1, 0], 3)
        self.assertEqual([metrics[key] for key in ("tp", "fp", "tn", "fn")], [1, 1, 1, 1])
        self.assertEqual([metrics[key] for key in ("precision", "recall", "f1")], [0.5, 0.5, 0.5])
        self.assertEqual(metrics["blocking_misses"], 1)
        self.assertEqual(metrics["classifier_misses_on_candidates"], 1)
        self.assertEqual(metrics["end_to_end_direct_pair_recall"], 1 / 3)
        self.assertEqual(metrics["end_to_end_missed_pairs"], 2)

    def test_undefined_metrics_and_empty_predictions(self):
        for targets, predictions in (([], []), ([0], [0])):
            metrics = tm.evaluate_predictions(targets, predictions, 0)
            for name in ("precision", "recall", "f1", "end_to_end_direct_pair_recall"):
                self.assertIsNone(metrics[name])
        metrics = tm.evaluate_predictions([1], [0], 2)
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)

    def test_different_related_entities_remain_negative_and_blocking_misses_stay_separate(self):
        pairs = [{"record_id_left": "a", "record_id_right": "b"},
                 {"record_id_left": "a", "record_id_right": "c"}]
        truth = {"a": "PARENT", "b": "SUBSIDIARY", "c": "PARENT", "d": "PARENT"}
        np.testing.assert_array_equal(tm.build_targets(pairs, truth), [0, 1])
        total, misses = tm.all_true_pairs_and_blocking_misses(truth, pairs)
        self.assertEqual(total, 3)
        self.assertEqual(len(misses), 2)
        self.assertEqual(len(pairs), 2)

    def test_truth_loader_filters_test_before_consuming_labels_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truth.csv"
            path.write_text("record_id,true_entity_id\n0001,0007\ntest-record\n", encoding="utf-8")
            self.assertEqual(tm.load_selected_truth(path, {"0001"}), {"0001": "0007"})


def write_pipeline_fixture(root):
    specifications = [
        ("t1", "E1", "train", "train-family"), ("t2", "E1", "train", "train-family"),
        ("t3", "E1", "train", "train-family"), ("t4", "E2", "train", "train-family"),
        ("t5", "E3", "train", "train-independent"),
        ("v1", "E4", "validation", "validation-family"), ("v2", "E4", "validation", "validation-family"),
        ("v3", "E4", "validation", "validation-family"), ("v4", "E5", "validation", "validation-family"),
        ("x1", "", "test", "test-family"), ("x2", "", "test", "test-family"),
    ]
    observed = []
    assignments = []
    truth = []
    for record_id, entity, split, group in specifications:
        row = {field: "" for field in pc.OBSERVED_FIELDS}
        row.update(record_id=record_id, source_system="SYNTHETIC_SOURCE", country="UK", city="Test City", postcode="00123")
        details = {"E1": ("Acme Manufacturing Ltd", "001", "10 High Street"),
                   "E2": ("Acme Technology Ltd", "002", "10 High Street"),
                   "E3": ("Birch Services Ltd", "003", "99 Low Road"),
                   "E4": ("Cedar Logistics Ltd", "004", "20 High Street"),
                   "E5": ("Cedar Technology Ltd", "005", "20 High Street"),
                   "": ("TEST OBSERVATIONS MUST NOT ENTER FEATURES", "999", "Test Only")}
        row["company_name"], row["registration_id"], row["address"] = details[entity]
        observed.append(row)
        assignments.append({"record_id": record_id, "true_entity_id": entity, "split_group_id": group,
                            "split": split, **pc.split_metadata(42, pc.SPLIT_RATIOS)})
        truth.append({"record_id": record_id, "true_entity_id": entity})
    pc.write_csv(root / "data/processed/cleaned_records.csv", pc.CLEANED_FIELDS, pc.clean_records(observed))
    pc.write_csv(root / "data/splits/record_splits.csv", pc.SPLIT_FIELDS, assignments)
    pc.write_csv(root / "data/ground_truth/record_entity_map.csv", pc.TRUTH_FIELDS, truth)
    for split, endpoints in {
        "train": [("t1", "t2"), ("t1", "t3"), ("t1", "t4"), ("t2", "t4"), ("t4", "t5")],
        "validation": [("v1", "v2"), ("v1", "v3"), ("v1", "v4"), ("v2", "v4")],
    }.items():
        pairs = [{"record_id_left": a, "record_id_right": b, "blocking_rules": "fuzzy_name_top_k"} for a, b in endpoints]
        pc.write_csv(root / f"data/processed/candidate_pairs_{split}.csv", pc.PAIR_FIELDS, pairs)
    pc.write_json(root / "reports/blocking_report.json", {
        "blocking_configuration": pc.asdict(pc.DEFAULT_BLOCKING_CONFIG),
        "fallback_configuration": pc.asdict(pc.DEFAULT_FALLBACK_CONFIG),
    })
    for relative in ("data/processed/candidate_pairs_test.csv", "data/ground_truth/entities.csv", "data/reference/parent_relationships.csv"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("MUST NOT OPEN", encoding="utf-8")


class PipelineChecks(unittest.TestCase):
    def test_training_scope_outputs_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_pipeline_fixture(root)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
            original_open = Path.open
            def guard(path, *args, **kwargs):
                if path.name in {"candidate_pairs_test.csv", "entities.csv", "parent_relationships.csv"}:
                    raise AssertionError("Forbidden input opened: " + path.name)
                return original_open(path, *args, **kwargs)
            with redirect_stdout(io.StringIO()), patch.object(Path, "open", guard), \
                    patch.object(tm, "fit_matcher", wraps=tm.fit_matcher) as fit, \
                    patch.object(tm, "build_feature_frame", wraps=tm.build_feature_frame) as build:
                report = tm.run_training(root)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(len(fit.call_args.args[0]), 5)
            np.testing.assert_array_equal(fit.call_args.args[1], [1, 1, 0, 0, 0])
            for call in build.call_args_list:
                self.assertFalse({"x1", "x2"} & set(call.args[1]))
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})
            self.assertEqual(set(report["evaluation"]), {"train", "validation"})
            for split in tm.EVALUATED_SPLITS:
                self.assertEqual(report["evaluation"][split]["blocking_misses"], 1)
            predictions = pc.read_csv(root / "reports/matching/validation_predictions.csv", tm.PREDICTION_FIELDS)
            self.assertEqual(len(predictions), 4)
            self.assertEqual(predictions[0]["registration_id_left"], "004")
            misses = pc.read_csv(root / "reports/matching/blocking_misses_train.csv", tm.BLOCKING_MISS_FIELDS)
            self.assertEqual(len(misses), 1)
            self.assertNotIn("logistic_score", misses[0])

    def test_family_isolation_and_candidate_endpoint_scope_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_pipeline_fixture(root)
            path = root / "data/splits/record_splits.csv"
            assignments = pc.read_csv(path, pc.SPLIT_FIELDS)
            assignments[-1]["split_group_id"] = "train-family"
            pc.write_csv(path, pc.SPLIT_FIELDS, assignments)
            with self.assertRaisesRegex(ValueError, "family crosses"):
                tm.load_stage3_inputs(root)
            write_pipeline_fixture(root)
            pairs_path = root / "data/processed/candidate_pairs_train.csv"
            pairs = pc.read_csv(pairs_path, pc.PAIR_FIELDS)
            pairs[0]["record_id_right"] = "v1"
            pc.write_csv(pairs_path, pc.PAIR_FIELDS, pairs)
            with self.assertRaisesRegex(ValueError, "outside its split"):
                tm.load_stage3_inputs(root)


if __name__ == "__main__":
    unittest.main()
