"""Stage 5 fixtures: connected components are predictions, not approved merges."""

import hashlib
import html
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

import build_clusters as bc
import cluster_views as cv
import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm
from test_matcher import write_pipeline_fixture


def records(*ids):
    return {record_id: {**{field: "" for field in pc.CLEANED_FIELDS}, "record_id": record_id,
                        "source_system": "SYNTHETIC_SOURCE", "company_name": "Observed " + record_id}
            for record_id in ids}


def pairs(*endpoints):
    return [{"record_id_left": left, "record_id_right": right, "blocking_rules": "fuzzy_name_top_k"}
            for left, right in endpoints]


def chain():
    return bc.build_duplicate_graph(records("a", "b", "c"), pairs(("a", "b"), ("b", "c")), [.9, .9], .5)


class GraphChecks(unittest.TestCase):
    def test_retains_singletons_with_no_candidates_or_no_predicted_match(self):
        graph = bc.build_duplicate_graph(records("a", "b", "c"), pairs(("a", "b")), [.1], .5)
        membership, clusters = bc.cluster_memberships(graph)
        self.assertEqual(set(graph), {"a", "b", "c"})
        self.assertEqual(len(clusters), 3)
        self.assertTrue(all(len(values) == 1 for values in clusters.values()))
        self.assertEqual(set(membership), set(graph))

    def test_cluster_ids_are_order_independent_and_membership_specific(self):
        self.assertEqual(bc.deterministic_cluster_id(["a", "b"]), bc.deterministic_cluster_id(["b", "a"]))
        self.assertNotEqual(bc.deterministic_cluster_id(["a", "b"]), bc.deterministic_cluster_id(["a,b"]))
        graph = chain()
        reversed_graph = nx.Graph()
        reversed_graph.add_nodes_from(reversed(list(graph)))
        reversed_graph.add_edges_from(reversed(list(graph.edges)))
        self.assertEqual(bc.cluster_memberships(graph), bc.cluster_memberships(reversed_graph))

    def test_saved_threshold_is_inclusive_and_edges_keep_score_and_provenance(self):
        graph = bc.build_duplicate_graph(records("a", "b", "c", "d"),
                                         pairs(("a", "b"), ("a", "c"), ("a", "d")), [.49999, .5, .7], .5)
        self.assertFalse(graph.has_edge("a", "b"))
        self.assertEqual(graph.edges["a", "c"]["model_score"], .5)
        self.assertEqual(graph.edges["a", "c"]["blocking_rules"], "fuzzy_name_top_k")
        self.assertEqual(graph.edges["a", "c"]["edge_type"], "direct_model_match")

    def test_transitive_chain_infers_membership_without_creating_an_edge(self):
        graph = bc.build_duplicate_graph(records("a", "b", "c"),
                                         pairs(("a", "b"), ("a", "c"), ("b", "c")), [.9, .1, .9], .5)
        membership, clusters = bc.cluster_memberships(graph)
        self.assertEqual(len(set(membership.values())), 1)
        self.assertFalse(graph.has_edge("a", "c"))
        summary = bc.cluster_summaries(graph, clusters, [])[0]
        self.assertEqual((summary["direct_edge_count"], summary["implied_pair_count"],
                          summary["inferred_without_direct_edge_count"]), (2, 3, 1))

    def test_split_isolation_rejects_outside_candidate_endpoints(self):
        with self.assertRaisesRegex(ValueError, "outside its split"):
            bc.build_duplicate_graph(records("a", "b"), pairs(("a", "z")), [.9], .5)
        train = bc.build_duplicate_graph(records("a", "b"), pairs(("a", "b")), [.9], .5)
        validation = bc.build_duplicate_graph(records("c", "d"), pairs(("c", "d")), [.9], .5)
        self.assertFalse(set(train) & set(validation))

    def test_invalid_scores_and_truth_columns_are_rejected(self):
        for scores in ([float("nan")], [1.1], []):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                bc.build_duplicate_graph(records("a", "b"), pairs(("a", "b")), scores, .5)
        forbidden = records("a")
        forbidden["a"]["true_entity_id"] = "LABEL"
        with self.assertRaisesRegex(ValueError, "columns"):
            bc.build_duplicate_graph(forbidden, [], [], .5)


class ReviewChecks(unittest.TestCase):
    def test_conflicting_nonempty_registration_ids_flag_but_do_not_change_membership(self):
        observed = records("a", "b", "c")
        observed["a"].update(country_normalized="gb", registration_id_normalized="001")
        observed["b"].update(country_normalized="gb", registration_id_normalized="002")
        observed["c"].update(country_normalized="gb", registration_id_normalized="")
        graph = chain()
        membership, clusters = bc.cluster_memberships(graph)
        before = list(graph.edges)
        flags = bc.registration_review_flags(clusters, observed)
        self.assertEqual(len(flags), 1)
        self.assertEqual(json.loads(flags[0]["conflicting_registration_ids"]), ["001", "002"])
        self.assertEqual(json.loads(flags[0]["supporting_record_ids"]), ["a", "b"])
        self.assertFalse(flags[0]["membership_changed"])
        self.assertEqual(bc.cluster_memberships(graph), (membership, clusters))
        self.assertEqual(list(graph.edges), before)

    def test_missing_country_missing_ids_and_different_countries_do_not_flag(self):
        observed = records("a", "b", "c")
        observed["a"].update(country_normalized="gb", registration_id_normalized="001")
        observed["b"].update(country_normalized="fr", registration_id_normalized="002")
        observed["c"].update(country_normalized="", registration_id_normalized="003")
        _, clusters = bc.cluster_memberships(chain())
        self.assertEqual(bc.registration_review_flags(clusters, observed), [])
        for row in observed.values():
            row.update(country_normalized="gb", registration_id_normalized="")
        self.assertEqual(bc.registration_review_flags(clusters, observed), [])


class MetricChecks(unittest.TestCase):
    def test_perfect_components_include_non_candidate_pairs(self):
        graph = chain()
        graph.add_node("d")
        metrics = bc.evaluate_clusters(graph, {"a": "E1", "b": "E1", "c": "E1", "d": "E2"})
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (3, 0, 0, 3))
        self.assertEqual((metrics["cluster_count"], metrics["singleton_count"], metrics["largest_cluster_size"]), (2, 1, 3))
        self.assertEqual(metrics["all_possible_pairs"], 6)
        self.assertEqual(metrics["pairwise_f1"], 1)

    def test_one_incorrect_edge_overmerges_groups_and_amplifies_false_pairs(self):
        graph = bc.build_duplicate_graph(records("a", "b", "c", "d"),
                                         pairs(("a", "b"), ("b", "c"), ("c", "d")), [.9, .51, .9], .5)
        metrics = bc.evaluate_clusters(graph, {"a": "E1", "b": "E1", "c": "E2", "d": "E2"})
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 4, 0))
        self.assertEqual(metrics["overmerged_cluster_count"], 1)
        self.assertEqual(metrics["fragmented_true_entity_count"], 0)
        self.assertEqual(metrics["pairwise_precision"], 1 / 3)
        self.assertEqual(metrics["pairwise_recall"], 1)
        self.assertEqual(metrics["pairwise_f1"], .5)

    def test_fragmentation_is_counted_per_true_entity(self):
        graph = bc.build_duplicate_graph(records("a", "b", "c"), pairs(("a", "b")), [.9], .5)
        metrics = bc.evaluate_clusters(graph, {"a": "E1", "b": "E1", "c": "E1"})
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 2))
        self.assertEqual(metrics["fragmented_true_entity_count"], 1)
        self.assertEqual(metrics["fragmented_true_entities"][0]["predicted_cluster_count"], 2)
        self.assertEqual(metrics["pairwise_recall"], 1 / 3)

    def test_zero_denominators_and_empty_graph(self):
        graph = nx.empty_graph(["a", "b"])
        metrics = bc.evaluate_clusters(graph, {"a": "E1", "b": "E2"})
        self.assertIsNone(metrics["pairwise_precision"])
        self.assertIsNone(metrics["pairwise_recall"])
        self.assertIsNone(metrics["pairwise_f1"])
        missing = bc.evaluate_clusters(graph, {"a": "E1", "b": "E1"})
        self.assertEqual(missing["pairwise_recall"], 0)
        self.assertEqual(missing["pairwise_f1"], 0)
        empty = bc.evaluate_clusters(nx.Graph(), {})
        self.assertEqual((empty["cluster_count"], empty["largest_cluster_size"]), (0, 0))

    def test_truth_coverage_is_required_for_evaluation_only(self):
        graph = chain()
        with self.assertRaisesRegex(ValueError, "truth coverage"):
            bc.evaluate_clusters(graph, {"a": "E1"})


def write_metadata(root):
    choice = {"chosen_model": "stage3_logistic_regression", "model_path": "models/matcher_pipeline.joblib",
              "feature_schema": list(pf.FEATURE_NAMES), "threshold": .5}
    pc.write_json(root / "models/next_stage_choice.json", choice)
    pc.write_json(root / "models/feature_names.json", list(pf.FEATURE_NAMES))
    pc.write_json(root / "models/matcher_config.json", {"prediction_threshold": .5})
    return choice


def fake_pipeline():
    return SimpleNamespace(feature_names_in_=np.array(pf.FEATURE_NAMES), named_steps={
        "scaler": SimpleNamespace(feature_names_in_=np.array(pf.FEATURE_NAMES)),
        "logistic_regression": SimpleNamespace(n_features_in_=21),
    })


class MetadataChecks(unittest.TestCase):
    def test_selected_feature_order_mismatch_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            choice = write_metadata(root)
            choice["feature_schema"] = list(reversed(pf.FEATURE_NAMES))
            pc.write_json(root / "models/next_stage_choice.json", choice)
            with self.assertRaisesRegex(ValueError, "ordered feature schema"):
                bc.load_selected_model(root)

    def test_threshold_metadata_mismatch_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            choice = write_metadata(root)
            choice["threshold"] = .6
            pc.write_json(root / "models/next_stage_choice.json", choice)
            with self.assertRaisesRegex(ValueError, "thresholds disagree"):
                bc.load_selected_model(root)

    def test_compatible_selected_model_loads_without_any_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_metadata(root)
            model = fake_pipeline()
            with patch.object(tm, "load_matcher", return_value=(model, .5)):
                loaded, threshold, _ = bc.load_selected_model(root)
            self.assertIs(loaded, model)
            self.assertEqual(threshold, .5)

    def test_loaded_scaler_feature_order_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_metadata(root)
            model = fake_pipeline()
            model.named_steps["scaler"].feature_names_in_ = np.array(list(reversed(pf.FEATURE_NAMES)))
            with patch.object(tm, "load_matcher", return_value=(model, .5)), self.assertRaisesRegex(ValueError, "scaler/classifier"):
                bc.load_selected_model(root)


class ExternalResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in {"script", "img", "iframe"} and values.get("src"):
            self.resources.append(values["src"])
        if tag == "link" and values.get("href"):
            self.resources.append(values["href"])


class ViewChecks(unittest.TestCase):
    def test_duplicate_html_escapes_observed_text_and_is_self_contained(self):
        observed = records("a", "b", "c")
        payload = "</script><img src=x onerror=alert('SYNTHETIC_XSS_PAYLOAD')>"
        observed["a"]["company_name"] = payload
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicates.html"
            result = cv.render_duplicate_example(chain(), observed, path)
            document = path.read_text(encoding="utf-8")
            self.assertNotIn(payload, document)
            self.assertIn(html.escape(payload, quote=True), document)
            self.assertIn("Inferred same-component membership only", document)
            parser = ExternalResources()
            parser.feed(document)
            self.assertEqual(parser.resources, [])
            self.assertEqual((result["full_component_size"], result["displayed_direct_edges"]), (3, 2))

    def test_reference_view_uses_only_supplied_endpoints_and_parent_to_child_arrows(self):
        relationships = [{"parent_entity_id": "P", "child_entity_id": "C", "relationship_source": "SYNTHETIC_SOURCE"}]
        self.assertEqual(set(cv.build_reference_graph(relationships).edges), {("P", "C")})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parents, names, output = root / "parents.csv", root / "names.csv", root / "reference.html"
            pc.write_csv(parents, pc.PARENT_FIELDS, relationships)
            pc.write_csv(names, ("true_entity_id", "company_name"), [
                {"true_entity_id": "P", "company_name": "Parent <Display>"},
                {"true_entity_id": "C", "company_name": "Child Ltd"},
                {"true_entity_id": "UNRELATED", "company_name": "UNRELATED_CANONICAL_MUST_NOT_BE_ADDED"},
            ])
            result = cv.render_reference_hierarchy(parents, names, output)
            document = output.read_text(encoding="utf-8")
            self.assertIn(cv.HIERARCHY_LABEL, document)
            self.assertIn("Parent &lt;Display&gt;", document)
            self.assertNotIn("UNRELATED_CANONICAL_MUST_NOT_BE_ADDED", document)
            self.assertEqual((result["entity_nodes"], result["supplied_relationship_edges"]), (2, 1))
            self.assertFalse(result["uses_record_truth_mapping"])
            parser = ExternalResources()
            parser.feed(document)
            self.assertEqual(parser.resources, [])
            with patch.object(cv, "write_view") as write:
                cv.render_reference_hierarchy(parents, names, output)
            arrow = write.call_args.args[3].layout.annotations[0]
            self.assertTrue(arrow.showarrow)
            self.assertGreater(arrow.ay, arrow.y)  # Root is above child; arrow points downward.

    def test_invalid_supplied_relationships_do_not_become_inferred_edges(self):
        relationship = {"parent_entity_id": "P", "child_entity_id": "C", "relationship_source": "SYNTHETIC"}
        reverse = {**relationship, "parent_entity_id": "C", "child_entity_id": "P"}
        with self.assertRaisesRegex(ValueError, "cycle"):
            cv.build_reference_graph([relationship, reverse])


class PipelineChecks(unittest.TestCase):
    def test_graphs_are_fixed_before_truth_and_no_training_or_retrieval_is_called(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_pipeline_fixture(root)
            parents = [{"parent_entity_id": "E1", "child_entity_id": "E2", "relationship_source": "SYNTHETIC"}]
            pc.write_csv(root / "data/reference/parent_relationships.csv", pc.PARENT_FIELDS, parents)
            pc.write_csv(root / "data/ground_truth/entities.csv", ("true_entity_id", "company_name"), [
                {"true_entity_id": "E1", "company_name": "Parent Ltd"}, {"true_entity_id": "E2", "company_name": "Child Ltd"}])
            paths = [path for path in root.rglob("*") if path.is_file()]
            manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            pc.write_json(root / "reports/clustering/preserved_inputs.json", {"sha256": manifest})
            original_truth = tm.load_selected_truth
            def truth_after_graphs(path, ids):
                for split in ("train", "validation"):
                    self.assertTrue((root / f"outputs/duplicate_clusters/{split}/cluster_membership.csv").is_file())
                    self.assertTrue((root / f"outputs/duplicate_clusters/{split}/review_flags.csv").is_file())
                return original_truth(path, ids)
            choice = {"chosen_model": "stage3_logistic_regression", "model_path": "models/matcher_pipeline.joblib"}
            with redirect_stdout(io.StringIO()), \
                    patch.object(bc, "load_selected_model", return_value=(object(), .5, choice)), \
                    patch.object(tm, "predict_logistic", side_effect=[(np.array([.9, .9, .1, .1, .1]), None), (np.array([.9, .9, .1, .1]), None)]), \
                    patch.object(tm, "load_selected_truth", side_effect=truth_after_graphs), \
                    patch.object(tm, "fit_matcher", side_effect=AssertionError("No fitting allowed")), \
                    patch.object(pc, "generate_revised_candidate_pairs", side_effect=AssertionError("No retrieval allowed")), \
                    patch.object(cv, "render_duplicate_example", return_value={"fixture": True}), \
                    patch.object(cv, "render_reference_hierarchy", return_value={"fixture": True}) as reference:
                report = bc.run_clustering(root)
            self.assertEqual(set(report["evaluation"]), {"train", "validation"})
            self.assertEqual((report["evaluation"]["train"]["records"], report["evaluation"]["validation"]["records"]), (5, 4))
            self.assertEqual(report["evaluation"]["validation"]["pairwise_recall"], 1)
            self.assertEqual(report["evaluation"]["validation"]["direct_predicted_edges"], 2)
            self.assertEqual(reference.call_args.args, (root / "data/reference/parent_relationships.csv",
                                                       root / "data/ground_truth/entities.csv",
                                                       root / "outputs/views/reference_corporate_hierarchy.html"))
            bc.verify_preserved_inputs(root)


if __name__ == "__main__":
    unittest.main()
