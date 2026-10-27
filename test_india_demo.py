"""Separate Indian-demo data/prediction checks. Never evaluate new accuracy."""

import copy
import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

import build_india_demo as build
import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm
from portfolio_ui import artifacts as ad, india


ROOT = Path(__file__).resolve().parent


class DemoFixtureChecks(unittest.TestCase):
    def test_24_records_six_verified_names_fake_ids_indian_cities_and_synthetic_pins(self):
        records, truth, companies, relationships, examples = build.make_demo_records()
        pc.validate_columns(records, pc.OBSERVED_FIELDS, "Demo")
        pc.validate_record_ids(records, "Demo")
        self.assertEqual((len(records), len(companies)), (24, 6))
        self.assertEqual(set(Counter(row["true_entity_id"] for row in truth).values()), {4})
        self.assertEqual({row["company_name"] for row in companies}, {
            "State Bank of India", "HDFC Bank Limited", "ICICI Bank Limited", "Axis Bank Limited",
            "Bajaj Finance Limited", "Bajaj Finserv Limited"})
        self.assertTrue(all(not row["registration_id"] or row["registration_id"].startswith("DEMO-IN-") for row in records))
        self.assertTrue(all(not row["postcode"] or re.fullmatch(r"\d{6}", row["postcode"]) for row in records))
        self.assertEqual({row["city"].casefold() for row in records}, {"mumbai", "hyderabad", "ahmedabad", "pune"})
        self.assertTrue(all(row["source_system"].startswith("DEMO_") for row in records))
        self.assertTrue(all("true_entity_id" not in row for row in records))

    def test_generation_is_reproducible_and_contains_typos_abbreviations_and_missing_values(self):
        a, truth_a, *_ = build.make_demo_records()
        b, truth_b, *_ = build.make_demo_records()
        self.assertEqual(a, b)
        self.assertEqual(truth_a, truth_b)
        names = {row["company_name"] for row in a}
        self.assertTrue({"SBI", "HDFC Bnak Ltd.", "Bajaj Fianance Ltd.", ""} <= names)
        self.assertTrue(any(not row["address"] for row in a))
        self.assertTrue(any(not row["registration_id"] for row in a))
        self.assertTrue(any("Rd." in row["address"] for row in a))

    def test_prediction_boundary_uses_actual_observations_and_existing_functions_without_truth(self):
        records, truth, *_ = build.make_demo_records()
        observed = copy.deepcopy(records)
        captured = []
        def predict(_pipeline, frame, threshold):
            pf.validate_feature_frame(frame)
            captured.append(frame.copy())
            return np.full(len(frame), .1), np.zeros(len(frame), dtype=int)
        with patch.object(tm, "predict_logistic", side_effect=predict), \
                patch.object(tm, "fit_matcher", side_effect=AssertionError("Never train")), \
                patch.object(tm, "load_selected_truth", side_effect=AssertionError("Never read truth for predictions")), \
                patch.object(pc, "read_csv", side_effect=AssertionError("Prediction helper must not open label/reference files")):
            result = build.predict_observed_records(records, object(), .5)
        self.assertEqual(records, observed)
        self.assertEqual(len(captured), 1)
        expected_cleaned = pc.clean_records(records)
        self.assertEqual(result["cleaned"], expected_cleaned)
        self.assertEqual(set(result["graph"]), {row["record_id"] for row in records})
        self.assertEqual(len(result["clusters"]), len(records))  # Fake model says no direct matches; retain all singletons.
        self.assertGreater(len(result["pairs"]), 0)
        for index, pair in enumerate(result["pairs"]):
            by_id = {row["record_id"]: row for row in expected_cleaned}
            left = {field: by_id[pair["record_id_left"]][field] for field in pf.NORMALIZED_FIELDS}
            right = {field: by_id[pair["record_id_right"]][field] for field in pf.NORMALIZED_FIELDS}
            self.assertEqual(captured[0].iloc[index].to_dict(), pf.build_pair_features(left, right))

    def test_extra_truth_columns_are_rejected_at_prediction_boundary(self):
        records, *_ = build.make_demo_records()
        records[0]["true_entity_id"] = "FORBIDDEN"
        with self.assertRaisesRegex(ValueError, "unexpected or missing columns"):
            build.predict_observed_records(records, object(), .5)


class SavedIndiaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ad.load_catalog(ROOT)
        cls.demo = india.load_demo(ROOT, cls.catalog)

    def test_new_prediction_provenance_keeps_frozen_settings_and_has_no_accuracy_metrics(self):
        run = build.verify_run(ROOT)
        self.assertEqual(run["batch_prediction_runs"], 1)
        self.assertEqual(run["model_sha256"], self.catalog.freeze["selected_model_sha256"])
        self.assertEqual(run["threshold"], self.catalog.choice["threshold"])
        self.assertEqual(run["feature_schema"], list(pf.FEATURE_NAMES))
        self.assertEqual(run["blocking_configuration"], self.catalog.freeze["blocking"])
        self.assertIn("NOT EVALUATED", run["performance"])
        self.assertFalse(run["truth_used_for_predictions"])
        self.assertFalse(run["reference_relationships_used_for_predictions"])
        self.assertFalse({"precision", "recall", "f1", "accuracy"} & set(run))
        self.assertEqual(run["builder_sha256"], hashlib.sha256((ROOT / "build_india_demo.py").read_bytes()).hexdigest())

    def test_no_original_record_ids_or_old_memberships_are_relabelled(self):
        old = pc.read_csv(ROOT / "data/raw/client_records.csv", pc.OBSERVED_FIELDS)
        ids = set(self.demo.records)
        self.assertFalse(ids & {row["record_id"] for row in old})
        pairs = pc.read_csv(ROOT / build.OUTPUT_DIR / "candidate_pairs.csv", pc.PAIR_FIELDS)
        pc.validate_candidate_pairs(pairs, ids)
        self.assertEqual({row["record_id"] for row in self.demo.bundle["members"]}, ids)
        predictions = pc.read_csv(ROOT / build.OUTPUT_DIR / "candidate_predictions.csv", build.PREDICTION_FIELDS)
        self.assertEqual(len(predictions), len(pairs))
        self.assertTrue(all("true_entity_id" not in row and "target" not in row for row in predictions))

    def test_filtering_shows_complete_groups_without_changing_memberships(self):
        before = copy.deepcopy(self.demo.bundle)
        selected = india.filtered_groups(self.demo, company="HDFC Bank Limited", source="DEMO_CORE")
        self.assertTrue(selected)
        for group in selected:
            self.assertTrue(any(row["company_name"] == "HDFC Bank Limited" and row["source_system"] == "DEMO_CORE"
                                for row in self.demo.bundle["by_cluster"][group]))
        self.assertEqual(before, self.demo.bundle)
        self.assertEqual(len(india.filtered_groups(self.demo, size_filter="Single-record groups")), self.demo.run["counts"]["singletons"])

    def test_guided_examples_use_real_record_pairs_and_truth_only_for_expected_outcomes(self):
        self.assertEqual(tuple(example["label"] for example in self.demo.examples), build.SCENARIO_LABELS)
        for example in self.demo.examples:
            self.assertIn(example["record_id_left"], self.demo.records)
            self.assertIn(example["record_id_right"], self.demo.records)
            self.assertEqual(india.educational_expectation(self.demo, example), example["expected_outcome"])
        self.assertEqual([row["expected_outcome"] for row in self.demo.examples],
                         ["same legal entity", "same legal entity", "different legal entities", "different legal entities"])

    def test_public_reference_sources_and_relationship_dates_are_saved(self):
        self.assertEqual(len(self.demo.companies), 6)
        for company in self.demo.companies:
            self.assertTrue(company["source_url"].startswith("https://"))
            self.assertEqual(company["retrieved_on"], "2026-10-26")
            self.assertTrue(company["verification_note"].startswith("Official"))
        graph, names = india.reference_graph(self.demo)
        self.assertEqual(len(graph), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        edge = self.demo.relationships[0]
        self.assertEqual((names[edge["parent_entity_id"]], names[edge["child_entity_id"]]), ("Bajaj Finserv Limited", "Bajaj Finance Limited"))
        self.assertEqual(edge["source_url"], "https://www.aboutbajajfinserv.com/finance-about-us")
        self.assertEqual(edge["retrieved_on"], "2026-10-26")

    def test_observed_group_names_are_not_ground_truth_company_assignments(self):
        for group in india.filtered_groups(self.demo):
            label = india.group_label(self.demo, group)
            observed_names = [row["company_name"] for row in self.demo.bundle["by_cluster"][group] if row["company_name"]]
            self.assertTrue(any(name in label for name in observed_names) if observed_names else "Company name missing" in label)

    def test_original_frozen_files_are_byte_identical(self):
        manifest = json.loads((ROOT / ad.FREEZE_DIR / "freeze_manifest.json").read_text())
        for relative, expected in manifest["protected_sha256"].items():
            self.assertEqual(ad.digest(ROOT / relative), expected, relative)

    def test_saved_demo_cannot_be_overwritten_by_repeating_builder(self):
        with patch.object(build, "predict_observed_records", side_effect=AssertionError("Must not score another batch")):
            with self.assertRaisesRegex(ValueError, "already exists"):
                build.build_demo(ROOT)


if __name__ == "__main__":
    unittest.main()
