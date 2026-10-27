"""Focused Stage 2 checks using tiny synthetic fixtures, never test-set metrics."""

import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import prepare_candidates as pc


def observed(record_id: str, **overrides: str) -> pc.Row:
    row = {field: "" for field in pc.OBSERVED_FIELDS}
    row.update({"record_id": record_id, "source_system": "SYNTHETIC_SOURCE"})
    row.update(overrides)
    return row


def split_fixture():
    # Three family members and three independent entities; E1 has two records.
    truth = [{"record_id": f"r{i}", "true_entity_id": entity}
             for i, entity in enumerate(("E1", "E1", "E2", "E3", "E4", "E5", "E6"))]
    records = [observed(row["record_id"]) for row in truth]
    parents = [
        {"child_entity_id": "E2", "parent_entity_id": "E1", "relationship_source": "SYNTHETIC"},
        {"child_entity_id": "E3", "parent_entity_id": "E2", "relationship_source": "SYNTHETIC"},
    ]
    return records, truth, parents


class CleaningChecks(unittest.TestCase):
    def test_unicode_suffix_address_and_original_values(self):
        original = observed(
            "0001", company_name="  ＡＣＭÉ,\u00a0LIMITED  ",
            address="Apt. 2, 10 High St.", country="U.K.",
        )
        cleaned = pc.clean_records([original])[0]
        for field in pc.OBSERVED_FIELDS:
            self.assertEqual(cleaned[field], original[field])
        self.assertEqual(cleaned["company_name_normalized"], "acmé ltd")
        self.assertEqual(pc.normalize_company_name("Acme\u0301 Ltd."), "acmé ltd")
        self.assertEqual(cleaned["address_normalized"], "flat 2 10 high street")
        self.assertEqual(cleaned["country_normalized"], "gb")

    def test_missing_values_and_literal_na(self):
        for value in (None, "", "  \t\n", "\u00a0"):
            self.assertEqual(pc.normalize_text(value), "")
        self.assertEqual(pc.normalize_country("NA"), "na")
        cleaned = pc.clean_records([observed("a"), observed("b")])
        self.assertEqual(pc.generate_candidate_pairs(cleaned), [])
        with self.assertRaises(TypeError):
            pc.normalize_compact(123)

    def test_csv_preserves_leading_zeros_and_does_not_infer_na(self):
        rows = [observed("0001", registration_id="00 012", postcode="00123", country="NA"),
                observed("0002", registration_id="12", postcode="123", country="NA")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observed.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=pc.OBSERVED_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            loaded = pc.load_observed_records(path)
        self.assertEqual(loaded, rows)
        cleaned = pc.clean_records(loaded)
        self.assertEqual(cleaned[0]["registration_id_normalized"], "00012")
        self.assertEqual(cleaned[0]["postcode_normalized"], "00123")
        self.assertEqual(pc.generate_candidate_pairs(cleaned), [])

    def test_input_rejects_missing_columns_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("record_id,source_system\na,SYNTHETIC\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                pc.load_observed_records(path)
        with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
            pc.clean_records([observed("a"), observed("a")])


class BlockingChecks(unittest.TestCase):
    def test_each_rule_skips_its_missing_required_values(self):
        cases = [
            {"country": "UK", "registration_id": ""},
            {"country": "", "registration_id": "00123"},
            {"country": "UK", "postcode": "", "company_name": "Acme Ltd"},
            {"country": "UK", "postcode": "00123", "company_name": ""},
            {"country": "", "postcode": "00123", "company_name": "Acme Ltd"},
            {"country": "UK", "city": "", "company_name": "Acme Ltd"},
            {"country": "UK", "city": "Test City", "company_name": ""},
            {"country": "", "city": "Test City", "company_name": "Acme Ltd"},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                rows = pc.clean_records([observed("a", **fields), observed("b", **fields)])
                self.assertEqual(pc.generate_candidate_pairs(rows), [])

    def test_tokens_exclude_generic_short_and_numeric_only_words(self):
        self.assertEqual(pc.informative_name_tokens("ltd limited plc services the 12345 ab"), ())
        self.assertEqual(pc.informative_name_tokens("acme acme ltd"), ("acme",))
        rows = pc.clean_records([
            observed("a", country="UK", city="Test City", company_name="Limited Services"),
            observed("b", country="UK", city="Test City", company_name="Ltd Services"),
        ])
        self.assertEqual(pc.generate_candidate_pairs(rows), [])
        self.assertEqual(pc.informative_name_tokens("acme", pc.BlockingConfig(minimum_token_length=5)), ())

    def test_union_deduplicates_all_rules_and_preserves_provenance(self):
        rows = pc.clean_records([
            observed(record_id, company_name="Acme Acme Limited", registration_id="00123",
                     country="UK", postcode="00123", city="Test City")
            for record_id in ("z", "a", "q")
        ])
        pairs = pc.generate_candidate_pairs(rows)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs, pc.generate_candidate_pairs(list(reversed(rows))))
        self.assertEqual({(row["record_id_left"], row["record_id_right"]) for row in pairs},
                         {("a", "q"), ("a", "z"), ("q", "z")})
        self.assertTrue(all(row["blocking_rules"] == "|".join(sorted(pc.RULE_NAMES)) for row in pairs))

    def test_same_source_and_different_source_are_both_allowed(self):
        for source in ("SYNTHETIC_SOURCE", "ANOTHER_SYNTHETIC_SOURCE"):
            rows = pc.clean_records([
                observed("a", country="UK", registration_id="00123"),
                observed("b", source_system=source, country="UK", registration_id="00123"),
            ])
            self.assertEqual(len(pc.generate_candidate_pairs(rows)), 1)

    def test_blocker_rejects_truth_and_family_columns(self):
        for forbidden in ("true_entity_id", "split_group_id", "parent_entity_id"):
            row = pc.clean_records([observed("a")])[0]
            row[forbidden] = "TRUTH_METADATA"
            with self.assertRaisesRegex(ValueError, "Blocking inputs"):
                pc.generate_candidate_pairs([row])

    def test_pair_validation_rejects_self_reversed_duplicate_and_cross_split_pairs(self):
        def pair(left, right):
            return {"record_id_left": left, "record_id_right": right,
                    "blocking_rules": pc.RULE_NAMES[0]}
        for pairs in ([pair("a", "a")], [pair("b", "a")],
                      [pair("a", "b"), pair("a", "b")], [pair("a", "outside")]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                pc.validate_candidate_pairs(pairs, {"a", "b"})


class SplitChecks(unittest.TestCase):
    def test_unequal_family_sizes_still_balance_records_and_entities(self):
        truth = [{"record_id": f"r{entity}_{copy}", "true_entity_id": f"E{entity}"}
                 for entity in range(300) for copy in range(1 + entity % 4)]
        parents = [{"parent_entity_id": f"E{start}", "child_entity_id": f"E{start + offset}",
                    "relationship_source": "SYNTHETIC"}
                   for start in range(0, 60, 5) for offset in range(1, 5)]
        groups = pc.build_split_groups(truth, parents)
        assignment = pc.create_split_assignments(truth, groups)
        for split, target in zip(pc.SPLIT_NAMES, pc.SPLIT_RATIOS):
            subset = [row for row in assignment if row["split"] == split]
            self.assertLess(abs(len(subset) / len(truth) - target), 0.03)
            self.assertLess(abs(len({row["true_entity_id"] for row in subset}) / 300 - target), 0.03)

    def test_family_components_and_reproducible_assignment(self):
        records, truth, parents = split_fixture()
        groups = pc.build_split_groups(truth, parents)
        self.assertEqual(groups["E1"], groups["E2"])
        self.assertEqual(groups["E2"], groups["E3"])
        self.assertEqual(len(set(groups.values())), 4)
        assignment = pc.create_split_assignments(truth, groups)
        self.assertEqual(assignment, pc.create_split_assignments(list(reversed(truth)), groups))
        pc.assert_split_isolation(assignment, records, truth, groups)

    def test_saved_assignment_is_reused_without_rewriting(self):
        records, truth, parents = split_fixture()
        groups = pc.build_split_groups(truth, parents)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "splits.csv"
            first, reused = pc.load_or_create_splits(path, records, truth, groups)
            self.assertFalse(reused)
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            second, reused = pc.load_or_create_splits(path, list(reversed(records)), truth, groups)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "never silently replaced"):
                pc.load_or_create_splits(path, records[:-1], truth, groups)
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_entity_record_and_corporate_family_isolation(self):
        records, truth, parents = split_fixture()
        groups = pc.build_split_groups(truth, parents)
        assignment = pc.create_split_assignments(truth, groups)
        for entity, expected in (("E1", "Entity crosses splits"), ("E2", "Corporate family crosses splits")):
            changed = [dict(row) for row in assignment]
            row = next(row for row in changed if row["true_entity_id"] == entity)
            row["split"] = next(split for split in pc.SPLIT_NAMES if split != row["split"])
            with self.assertRaisesRegex(ValueError, expected):
                pc.assert_split_isolation(changed, records, truth, groups)
        with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
            pc.assert_split_isolation(assignment + [assignment[0]], records, truth, groups)

    def test_changed_family_membership_and_invalid_parent_references_fail(self):
        records, truth, parents = split_fixture()
        groups = pc.build_split_groups(truth, parents)
        assignment = pc.create_split_assignments(truth, groups)
        new_link = {"parent_entity_id": "E3", "child_entity_id": "E4", "relationship_source": "SYNTHETIC"}
        merged = pc.build_split_groups(truth, parents + [new_link])
        with self.assertRaisesRegex(ValueError, "family membership has changed"):
            pc.assert_split_isolation(assignment, records, truth, merged)
        new_link["child_entity_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "Invalid parent relationship reference"):
            pc.build_split_groups(truth, parents + [new_link])


class FallbackChecks(unittest.TestCase):
    def test_supplied_defaults_and_disabled_fallback(self):
        config = pc.DEFAULT_FALLBACK_CONFIG
        self.assertEqual((config.company_name_top_k, config.company_name_min_score), (5, 80))
        self.assertEqual((config.address_top_k, config.address_min_score), (3, 85))
        rows = pc.clean_records([observed("a", company_name="Acme Ltd"),
                                 observed("b", company_name="Acme Limited")])
        disabled = pc.FallbackConfig(enabled=False)
        self.assertEqual(pc.generate_fallback_candidates(rows, disabled), [])
        self.assertEqual(pc.generate_revised_candidate_pairs(rows, fallback_config=disabled),
                         pc.generate_candidate_pairs(rows))
        for settings in ({"company_name_top_k": 0}, {"address_top_k": -1},
                         {"company_name_min_score": 101}, {"address_min_score": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                pc.FallbackConfig(**settings)

    def test_name_fallback_ignores_missing_or_different_geography(self):
        rows = pc.clean_records([
            observed("a", company_name="Acme Industries Limited", country="", city="First City"),
            observed("b", company_name="Acme Industries Ltd.", country="France", city="Other City"),
        ])
        self.assertEqual(pc.generate_candidate_pairs(rows), [])
        self.assertEqual(pc.generate_fallback_candidates(rows), [
            {"record_id_left": "a", "record_id_right": "b", "blocking_rules": "fuzzy_name_top_k"}
        ])

    def test_address_fallback_only_for_missing_name_queries_and_can_target_named_records(self):
        rows = pc.clean_records([
            observed("a", company_name="", address="Flat 2, 10 High St."),
            observed("b", company_name="Acme Ltd", address="Apt 2 10 High Street"),
        ])
        self.assertEqual(pc.generate_fallback_candidates(rows), [
            {"record_id_left": "a", "record_id_right": "b", "blocking_rules": "fuzzy_address_top_k"}
        ])
        named = pc.clean_records([observed("a", company_name="AAAAAA", address="10 High Street"),
                                  observed("b", company_name="ZZZZZZ", address="10 High Street")])
        self.assertEqual(pc.generate_fallback_candidates(named), [])

    def test_missing_text_is_never_scored(self):
        rows = pc.clean_records([observed("a"), observed("b", company_name="...", address="  ")])
        with patch.object(pc.fuzz, "WRatio", side_effect=AssertionError("Missing text was scored")):
            self.assertEqual(pc.generate_fallback_candidates(rows), [])
            self.assertEqual(pc.top_fuzzy_neighbors("a", "", {"b": ""}, 5, 80), [])
            self.assertEqual(pc.top_fuzzy_neighbors("a", "acme", {"a": "acme", "b": ""}, 5, 80), [])

    def test_cutoffs_are_inclusive_and_low_scores_are_excluded(self):
        for cutoff in (80, 85):
            with self.subTest(cutoff=cutoff), patch.object(pc.fuzz, "WRatio", side_effect=[cutoff, cutoff - 0.01]):
                self.assertEqual(pc.top_fuzzy_neighbors("a", "query", {"b": "first", "c": "second"}, 5, cutoff),
                                 [("b", cutoff)])

    def test_top_k_ties_are_deterministic_and_self_does_not_consume_a_slot(self):
        for top_k in (5, 3):
            choices = {record_id: "identical company" for record_id in ("z", "g", "f", "e", "d", "c", "b", "a")}
            expected = [(record_id, 100.0) for record_id in ("a", "b", "c", "d", "e")[:top_k]]
            self.assertEqual(pc.top_fuzzy_neighbors("z", "identical company", choices, top_k, 80), expected)
            self.assertEqual(pc.top_fuzzy_neighbors("z", "identical company", dict(reversed(list(choices.items()))),
                                                    top_k, 80), expected)
        rows = pc.clean_records([observed(record_id, company_name="Identical Ltd") for record_id in ("c", "b", "a")])
        self.assertEqual(pc.generate_fallback_candidates(rows), pc.generate_fallback_candidates(list(reversed(rows))))

    def test_union_deduplicates_reversed_pairs_and_retains_exact_and_fuzzy_provenance(self):
        exact = {"record_id_left": "a", "record_id_right": "b", "blocking_rules": pc.RULE_NAMES[0]}
        fuzzy = {"record_id_left": "b", "record_id_right": "a", "blocking_rules": "fuzzy_name_top_k"}
        self_pair = {**fuzzy, "record_id_left": "a"}
        union = pc.union_candidate_pairs([exact], [fuzzy, fuzzy, self_pair])
        self.assertEqual(union, [{**exact, "blocking_rules": pc.RULE_NAMES[0] + "|fuzzy_name_top_k"}])
        rows = pc.clean_records([observed(record_id, company_name="Acme Ltd", registration_id="001", country="UK")
                                 for record_id in ("a", "b")])
        revised = pc.generate_revised_candidate_pairs(rows)
        self.assertEqual(len(revised), 1)
        self.assertEqual(revised[0]["blocking_rules"], pc.RULE_NAMES[0] + "|fuzzy_name_top_k")

    def test_separate_split_searches_cannot_see_identical_outside_records(self):
        train = pc.clean_records([observed("train", company_name="Acme Ltd", address="10 High Street")])
        validation = pc.clean_records([observed("validation", company_name="Acme Ltd", address="10 High Street")])
        for subset in (train, validation):
            self.assertEqual(pc.generate_revised_candidate_pairs(subset), [])
        self.assertEqual(len(pc.generate_revised_candidate_pairs(train + validation)), 1)
        leaked = dict(train[0], true_entity_id="LABEL")
        with self.assertRaisesRegex(ValueError, "Fallback inputs"):
            pc.generate_fallback_candidates([leaked])


class DiagnosisChecks(unittest.TestCase):
    def test_diagnosis_counts_missing_fields_without_treating_them_as_agreements(self):
        rows = pc.clean_records([observed("a", company_name="Acme Ltd", address="10 High Street"),
                                 observed("b", company_name="Acme Limited", address="10 High St.")])
        truth = [{"record_id": record_id, "true_entity_id": "E1"} for record_id in ("a", "b")]
        result = pc.diagnose_missed_training_pairs(rows, truth, [])
        self.assertEqual(result["missed_true_duplicate_pairs"], 1)
        self.assertEqual(result["observed_field_state_counts"]["country"], {"both_missing": 1})
        self.assertEqual(result["observed_field_state_counts"]["address"], {"equal_nonempty": 1})
        self.assertTrue(all(result["missed_pair_details"][0]["failed_rule_reasons"].values()))
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "restricted to training"):
                pc.diagnose_missed_training_pairs(rows, truth, [], split=split)


class EvaluationChecks(unittest.TestCase):
    def test_complete_true_pair_denominator_and_no_pair_injection(self):
        truth = [{"record_id": record_id, "true_entity_id": entity}
                 for record_id, entity in (("a", "E1"), ("b", "E1"), ("c", "E1"), ("d", "E2"))]
        pairs = [{"record_id_left": "a", "record_id_right": "b", "blocking_rules": pc.RULE_NAMES[0]}]
        result = pc.evaluate_blocking(pairs, truth, "train")
        self.assertEqual(result["total_possible_pairs"], 6)
        self.assertEqual(result["all_true_duplicate_pairs"], 3)
        self.assertEqual(result["true_duplicate_pairs_retrieved"], 1)
        self.assertEqual(result["blocking_recall"], 1 / 3)
        self.assertEqual(result["missed_true_duplicate_pairs"], 2)
        self.assertEqual(result["candidate_match_count"], 1)
        self.assertEqual(result["candidate_non_match_count"], 0)
        self.assertEqual(len(pairs), 1)

    def test_zero_denominators_and_test_evaluation_guard(self):
        for truth in ([], [{"record_id": "a", "true_entity_id": "E1"}]):
            result = pc.evaluate_blocking([], truth, "validation")
            self.assertIsNone(result["pair_reduction"])
            self.assertIsNone(result["blocking_recall"])
        with self.assertRaisesRegex(ValueError, "reserved for final evaluation"):
            pc.evaluate_blocking([], [], "test")


class PipelineChecks(unittest.TestCase):
    def test_improvement_preserves_baseline_and_evaluates_each_allowed_split_once(self):
        records, truth, parents = split_fixture()
        records = [{**row, "company_name": "Acme Manufacturing Ltd"} for row in records]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative, fields, rows in (
                ("data/raw/client_records.csv", pc.OBSERVED_FIELDS, records),
                ("data/ground_truth/record_entity_map.csv", pc.TRUTH_FIELDS, truth),
                ("data/reference/parent_relationships.csv", pc.PARENT_FIELDS, parents),
            ):
                pc.write_csv(root / relative, fields, rows)
            # Freeze a tiny exact-only fixture, not the actual project's validation data.
            with redirect_stdout(io.StringIO()):
                baseline = pc.run_pipeline(root, fallback_config=pc.FallbackConfig(enabled=False))
            baseline.pop("fallback_configuration")
            archive = root / "reports/blocking_baseline"
            pc.write_json(archive / "blocking_report.json", baseline)
            artifacts = [archive / "blocking_report.json"]
            for split in pc.SPLIT_NAMES:
                destination = root / f"data/processed/baseline/candidate_pairs_{split}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / f"data/processed/candidate_pairs_{split}.csv", destination)
                artifacts.append(destination)
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            inputs = [root / relative for relative in (
                "data/raw/client_records.csv", "data/ground_truth/record_entity_map.csv",
                "data/reference/parent_relationships.csv", "data/splits/record_splits.csv",
            )]
            manifest = {
                "protected_input_sha256": {path.relative_to(root).as_posix(): digest(path) for path in inputs},
                "baseline_artifact_sha256": {path.relative_to(root).as_posix(): digest(path) for path in artifacts},
            }
            pc.write_json(archive / "manifest.json", manifest)
            frozen = {path: path.read_bytes() for path in inputs + artifacts + [archive / "manifest.json"]}
            split_mtime = (root / "data/splits/record_splits.csv").stat().st_mtime_ns
            with redirect_stdout(io.StringIO()), patch.object(pc, "evaluate_blocking", wraps=pc.evaluate_blocking) as evaluation:
                comparison = pc.run_improvement(root)
            self.assertEqual([call.args[2] for call in evaluation.call_args_list], ["train", "validation"])
            self.assertEqual(frozen, {path: path.read_bytes() for path in frozen})
            self.assertEqual(split_mtime, (root / "data/splits/record_splits.csv").stat().st_mtime_ns)
            self.assertEqual(set(comparison["comparison"]), {"train", "validation"})
            self.assertEqual(json.loads((root / "reports/blocking_comparison.json").read_text()), comparison)
            self.assertGreater(comparison["comparison"]["train"]["change"]["additional_candidates"], 0)
            for values in comparison["comparison"].values():
                for version in ("baseline", "revised"):
                    metrics = values[version]
                    self.assertEqual(metrics["candidate_pairs"],
                                     metrics["candidate_match_count"] + metrics["candidate_non_match_count"])
            inputs[0].write_bytes(inputs[0].read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "protected input changed"):
                pc.load_frozen_baseline(root, pc.DEFAULT_BLOCKING_CONFIG)

    def test_outputs_reuse_and_input_preservation_without_reading_canonical_details(self):
        records, truth, parents = split_fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative, fields, rows in (
                ("data/raw/client_records.csv", pc.OBSERVED_FIELDS, records),
                ("data/ground_truth/record_entity_map.csv", pc.TRUTH_FIELDS, truth),
                ("data/reference/parent_relationships.csv", pc.PARENT_FIELDS, parents),
            ):
                pc.write_csv(root / relative, fields, rows)
            # Deliberately invalid canonical content proves the pipeline does not read it.
            (root / "data/ground_truth/entities.csv").write_text("DO NOT READ", encoding="utf-8")
            def hashes():
                return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in root.rglob("*") if path.is_file()}
            inputs_before = hashes()
            with redirect_stdout(io.StringIO()):
                first = pc.run_pipeline(root)
                all_outputs_before = hashes()
                second = pc.run_pipeline(root)
            self.assertEqual(first, second)
            self.assertEqual(all_outputs_before, hashes())
            self.assertTrue(all(hashes()[name] == digest for name, digest in inputs_before.items()))
            self.assertEqual(set(first["blocking_evaluation"]), {"train", "validation"})
            self.assertEqual(json.loads((root / "reports/blocking_report.json").read_text()), second)
            self.assertEqual(len(pc.read_csv(root / "data/processed/cleaned_records.csv", pc.CLEANED_FIELDS)),
                             len(records))
            for split in pc.SPLIT_NAMES:
                self.assertEqual(pc.read_csv(root / f"data/processed/candidate_pairs_{split}.csv", pc.PAIR_FIELDS), [])


if __name__ == "__main__":
    unittest.main()
