"""Indian app AppTest checks. No training, benchmark evaluation or downloads."""

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import pair_features as pf
import prepare_candidates as pc
from portfolio_ui import artifacts as ad, india, presentation as ui, scoring
from portfolio_ui.screens import SECTIONS


ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"


def create_app():
    return AppTest.from_file(str(APP), default_timeout=35).run()


def compare_page():
    app = create_app()
    return app.radio(key="navigation").set_value(SECTIONS[1]).run()


class IndianAppChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ad.load_catalog(ROOT)
        cls.demo = india.load_demo(ROOT, cls.catalog)

    def assert_healthy(self, app):
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))

    def test_start_leads_with_real_dataset_example_and_try_button(self):
        app = create_app()
        self.assert_healthy(app)
        self.assertEqual(app.title[0].value, "Banking Client Entity Resolution")
        self.assertEqual(app.radio(key="navigation").value, "Start here")
        self.assertTrue(any("HDFC Bank Limited" == item.value for item in app.text))
        self.assertTrue(any("HDFC Bnak Ltd." == item.value for item in app.text))
        self.assertTrue(any("Original v1 benchmark details" in expander.label for expander in app.expander))
        app.button(key="try_example").click().run()
        self.assert_healthy(app)
        self.assertEqual(app.radio(key="navigation").value, SECTIONS[1])
        self.assertEqual(app.selectbox(key="guided_example").value, self.demo.examples[0]["scenario_id"])

    def test_all_five_sections_have_short_beginner_instructions(self):
        app = create_app()
        self.assertEqual(list(app.radio(key="navigation").options), list(SECTIONS))
        for section in SECTIONS:
            app.radio(key="navigation").set_value(section).run()
            self.assert_healthy(app)
            self.assertTrue(any(header.value == section for header in app.header))
            text = " ".join(item.value for item in app.markdown)
            for label in ("What this page does", "What to select", "How to interpret the result"):
                self.assertIn(label, text)

    def test_dropdowns_use_actual_indian_record_labels_and_exclude_a_from_b(self):
        app = compare_page()
        self.assert_healthy(app)
        self.assertEqual(app.radio(key="comparison_mode").value, "Choose existing records")
        self.assertEqual(set(app.selectbox(key="record_a").options), {india.record_label(row) for row in self.demo.records.values()})
        selected = app.selectbox(key="record_a").value
        self.assertNotIn(india.record_label(self.demo.records[selected]), app.selectbox(key="record_b").options)
        self.assertEqual(len(app.selectbox(key="record_b").options), len(self.demo.records) - 1)
        self.assertFalse(app.text_input)  # Original details are read-only tables.
        self.assertEqual([button.label for button in app.button], ["Compare selected records"])

    def test_selection_autofills_actual_fields_and_clears_previous_result(self):
        app = compare_page()
        app.button(key="compare_selected").click().run()
        self.assert_healthy(app)
        self.assertTrue(app.metric)
        chosen = self.demo.examples[1]["record_id_left"]
        app.selectbox(key="record_a").select(chosen).run()
        self.assert_healthy(app)
        expected = self.demo.records[chosen]
        table = app.dataframe[0].value
        displayed = dict(zip(table["Field"], table["Selected value"]))
        self.assertEqual(displayed["Company name"], expected["company_name"])
        self.assertEqual(displayed["Fake demo identifier"], expected["registration_id"])
        self.assertEqual(displayed["Synthetic PIN code"], expected["postcode"])
        self.assertFalse(app.metric)
        self.assertNotEqual(app.selectbox(key="record_a").value, app.selectbox(key="record_b").value)

    def test_changing_a_to_the_old_b_never_leaves_identical_selections(self):
        app = compare_page()
        old_b = app.selectbox(key="record_b").value
        app.selectbox(key="record_a").select(old_b).run()
        self.assert_healthy(app)
        self.assertEqual(app.selectbox(key="record_a").value, old_b)
        self.assertNotEqual(app.selectbox(key="record_b").value, old_b)

    def test_all_guided_scenarios_select_saved_pairs_and_keep_expectations_separate(self):
        app = compare_page()
        for example in self.demo.examples:
            app.selectbox(key="guided_example").select(example["scenario_id"]).run()
            self.assert_healthy(app)
            self.assertEqual(app.selectbox(key="record_a").value, example["record_id_left"])
            self.assertEqual(app.selectbox(key="record_b").value, example["record_id_right"])
            self.assertTrue(any("Educational expectation from simulation labels" in item.value and example["expected_outcome"] in item.value for item in app.info))
            self.assertFalse(app.metric)  # Selecting a teaching label does not predict or override anything.

    def test_scenario_change_and_manual_record_change_clear_stale_scores(self):
        app = compare_page()
        app.selectbox(key="guided_example").select(self.demo.examples[0]["scenario_id"]).run()
        app.button(key="compare_selected").click().run()
        self.assert_healthy(app)
        self.assertTrue(app.metric)
        app.selectbox(key="guided_example").select(self.demo.examples[1]["scenario_id"]).run()
        self.assertFalse(app.metric)
        app.button(key="compare_selected").click().run()
        self.assertTrue(app.metric)
        app.selectbox(key="record_b").select(self.demo.examples[0]["record_id_right"]).run()
        self.assertFalse(app.metric)
        self.assertIsNone(app.selectbox(key="guided_example").value)

    def test_live_score_receives_the_actual_indian_observations_not_v1_or_canonical_fields(self):
        app = compare_page()
        import portfolio_ui.screens as screens
        with patch.object(screens, "score_records", wraps=scoring.score_records) as score:
            app.button(key="compare_selected").click().run()
        self.assert_healthy(app)
        self.assertEqual(score.call_count, 1)
        left = self.demo.records[app.selectbox(key="record_a").value]
        right = self.demo.records[app.selectbox(key="record_b").value]
        self.assertEqual(score.call_args.args[2], {field: left[field] for field in scoring.FIELDS})
        self.assertEqual(score.call_args.args[3], {field: right[field] for field in scoring.FIELDS})
        self.assertTrue(any(metric.label == "Uncalibrated model score" for metric in app.metric))
        self.assertTrue(any("bypasses candidate blocking" in item.value for item in app.caption))

    def test_advanced_copy_edits_are_temporary_and_clear_old_decisions(self):
        app = compare_page()
        app.button(key="compare_selected").click().run()
        self.assertTrue(app.metric)
        app.radio(key="comparison_mode").set_value("Advanced: edit a copy").run()
        self.assert_healthy(app)
        self.assertFalse(app.metric)
        saved = dict(self.demo.records[app.selectbox(key="record_a").value])
        app.text_input(key="left_company_name").set_value("A temporary edited demo name").run()
        app.button(key="compare_selected").click().run()
        self.assert_healthy(app)
        self.assertTrue(app.metric)
        app.text_input(key="left_company_name").set_value("Another temporary name").run()
        self.assertFalse(app.metric)
        self.assertEqual(self.demo.records[app.selectbox(key="record_a").value], saved)
        app.radio(key="comparison_mode").set_value("Choose existing records").run()
        self.assertEqual(app.dataframe[0].value["Selected value"].iloc[0], saved["company_name"])

    def test_empty_advanced_copy_is_rejected(self):
        app = compare_page()
        app.radio(key="comparison_mode").set_value("Advanced: edit a copy").run()
        for field in scoring.FIELDS:
            app.text_input(key="left_" + field).set_value(" ")
        app.button(key="compare_selected").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Record A is empty" in error.value for error in app.error))
        self.assertFalse(app.metric)

    def test_incorrect_model_result_is_not_overridden_by_teaching_truth(self):
        app = compare_page()
        app.selectbox(key="guided_example").select(self.demo.examples[0]["scenario_id"]).run()
        from portfolio_ui import cache
        pipeline, threshold = ad.load_frozen_model(self.catalog)
        example = self.demo.examples[0]
        left = {field: self.demo.records[example["record_id_left"]][field] for field in scoring.FIELDS}
        right = {field: self.demo.records[example["record_id_right"]][field] for field in scoring.FIELDS}
        fixture_result = scoring.score_records(pipeline, threshold, left, right)
        fixture_result.update(score=.01, prediction=False)
        with patch("portfolio_ui.screens.score_records", return_value=fixture_result):
            app.button(key="compare_selected").click().run()
        self.assert_healthy(app)
        self.assertTrue(any("same legal entity" in item.value for item in app.info))
        self.assertTrue(any(item.value == "Predicted non-match" for item in app.info))
        self.assertFalse(any(item.value == "Predicted match" for item in app.success))

    def test_group_dropdowns_are_dataset_driven_and_filters_keep_whole_groups(self):
        app = create_app()
        app.radio(key="navigation").set_value(SECTIONS[2]).run()
        self.assert_healthy(app)
        names = {row["company_name"] or "(company name missing)" for row in self.demo.records.values()}
        self.assertEqual(set(app.selectbox(key="group_company").options) - {"All observed names"}, names)
        self.assertEqual(set(app.selectbox(key="group_source").options) - {"All demo systems"}, {row["source_system"] for row in self.demo.records.values()})
        app.selectbox(key="group_company").select("HDFC Bank Limited").run()
        self.assert_healthy(app)
        selected = app.selectbox(key="selected_group").value
        original_count = len(self.demo.bundle["by_cluster"][selected])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Records in this group"], str(original_count))
        self.assertTrue(app.get("download_button"))
        self.assertTrue(any("Full technical IDs" in expander.label for expander in app.expander))

    def test_singleton_and_review_filters_remain_available(self):
        app = create_app()
        app.radio(key="navigation").set_value(SECTIONS[2]).run()
        app.selectbox(key="group_size").select("Single-record groups").run()
        self.assert_healthy(app)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Records in this group"), "1")
        app.selectbox(key="group_size").select("All groups").run()
        app.checkbox(key="group_review_only").check().run()
        self.assert_healthy(app)
        self.assertTrue(app.warning)

    def test_hierarchy_defaults_to_verified_bajaj_reference_not_uk_or_predictions(self):
        app = create_app()
        app.radio(key="navigation").set_value(SECTIONS[3]).run()
        self.assert_healthy(app)
        self.assertIn("Bajaj Finserv Limited", app.selectbox(key="hierarchy_family").options[0])
        self.assertTrue(any("not predictions" in item.value and "not an exhaustive" in item.value for item in app.info))
        self.assertTrue(any("Bajaj Finserv Limited → Bajaj Finance Limited" == item.value for item in app.markdown))
        self.assertTrue(any("2026-10-26" in item.value for item in app.caption))

    def test_original_benchmark_is_explicitly_separate_and_loaded_from_report(self):
        app = create_app()
        self.assert_healthy(app)
        self.assertTrue(any("Original v1 benchmark results" in item.value and "not Indian-demo performance" in item.value for item in app.warning))
        table = app.dataframe[0].value
        saved = self.catalog.report["metrics"]["candidate_classification"]["logistic_regression"]
        self.assertEqual(table.loc[table["Method"] == "Logistic regression", "Candidate F1"].iloc[0], ui.percent(saved["f1"]))

    def test_missing_original_model_stops_without_regenerating(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ad.RUNTIME_PATHS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            (root / "models/matcher_pipeline.joblib").unlink()
            with patch.object(ad, "PROJECT_ROOT", root):
                app = create_app()
            self.assertFalse(app.exception)
            self.assertTrue(any("Missing required artifact" in error.value for error in app.error))
            self.assertFalse((root / "models/matcher_pipeline.joblib").exists())


class AdapterChecks(unittest.TestCase):
    def test_scoring_symmetry_missing_evidence_exclusions_and_leading_zeros(self):
        catalog = ad.load_catalog(ROOT)
        pipeline, threshold = ad.load_frozen_model(catalog)
        values = {field: "" for field in scoring.FIELDS}
        values.update(country="India", company_name="Demo Test Limited", registration_id="DEMO-IN-00123", postcode="001234")
        other = {**values, "company_name": "Demo Test Ltd"}
        a = scoring.score_records(pipeline, threshold, values, other)
        b = scoring.score_records(pipeline, threshold, other, values)
        self.assertEqual(a["score"], b["score"])
        self.assertEqual(a["features"], b["features"])
        self.assertEqual(a["left_normalized"]["postcode_normalized"], "001234")
        self.assertEqual(a["features"]["address_ratio"], 0)
        self.assertEqual(a["features"]["address_both_missing"], 1)
        with self.assertRaises(ValueError):
            scoring.score_records(pipeline, threshold, {**values, "true_entity_id": "truth"}, other)

    def test_downloads_and_figures_use_observed_fields_and_escape_text(self):
        rows = [{field: "" for field in pc.OBSERVED_FIELDS} for _ in range(2)]
        for row, record_id in zip(rows, ("a", "b")):
            row.update(record_id=record_id, company_name='<script>alert("x")</script>', postcode="001234")
        edges = [{"record_id_left": "a", "record_id_right": "b", "model_score": ".9", "blocking_rules": "fuzzy_name_top_k"}]
        encoded = json.dumps(ui.duplicate_figure(rows, edges, compact_ids=True).to_plotly_json())
        self.assertNotIn("<script>", encoded)
        self.assertIn("&lt;script&gt;", encoded)
        import csv
        decoded = list(csv.DictReader(io.StringIO(ui.download_records(rows).decode("utf-8-sig"))))
        self.assertEqual(set(decoded[0]), set(pc.OBSERVED_FIELDS))
        self.assertEqual(decoded[0]["postcode"], "001234")

    def test_no_optional_encoder_needed_for_all_sections_and_live_scoring(self):
        script = r'''
import importlib.abc,sys
class NoEncoder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','sentence_transformers','transformers','huggingface_hub'}:
            raise ImportError('Optional encoder intentionally blocked')
sys.meta_path.insert(0,NoEncoder())
from pathlib import Path
from streamlit.testing.v1 import AppTest
from portfolio_ui.screens import SECTIONS
app=AppTest.from_file(str(Path.cwd()/'app.py'),default_timeout=35).run()
for section in SECTIONS:
    app.radio(key='navigation').set_value(section).run()
    assert not app.exception and not app.error,(section,str(app.exception),str(app.error))
app.radio(key='navigation').set_value(SECTIONS[1]).run()
app.button(key='compare_selected').click().run()
assert not app.error and not app.exception and app.metric
assert not {'torch','sentence_transformers'} & set(sys.modules)
print('PASS: Indian app and live scoring without optional encoder imports')
'''
        process = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=75)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("PASS:", process.stdout)


if __name__ == "__main__":
    unittest.main()
