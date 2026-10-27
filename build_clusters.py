"""Stage 5: predicted duplicate components and a separate supplied hierarchy.

No fitting, threshold search, retrieval changes, approved merges, or test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np

import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm


PROJECT_ROOT = Path(__file__).resolve().parent
MEMBERSHIP_FIELDS = ("cluster_id", "cluster_size", *pc.CLEANED_FIELDS)
EDGE_FIELDS = ("cluster_id", "record_id_left", "record_id_right", "model_score", "blocking_rules", "edge_type")
FLAG_FIELDS = ("cluster_id", "country_normalized", "conflicting_registration_ids", "supporting_record_ids",
               "review_reason", "membership_changed")
SUMMARY_FIELDS = ("cluster_id", "record_count", "direct_edge_count", "implied_pair_count",
                  "inferred_without_direct_edge_count", "review_required")


def verify_preserved_inputs(root):
    manifest = json.loads((root / "reports/clustering/preserved_inputs.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["sha256"].items():
        pc.require(hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest,
                   f"Protected prior-stage artifact changed: {relative}")
    return len(manifest["sha256"])


def load_selected_model(root):
    """Check choice, saved metadata, runtime allowlist, scaler, and fitted model."""
    choice = json.loads((root / "models/next_stage_choice.json").read_text(encoding="utf-8"))
    names = json.loads((root / "models/feature_names.json").read_text(encoding="utf-8"))
    config = json.loads((root / "models/matcher_config.json").read_text(encoding="utf-8"))
    pc.require(choice["chosen_model"] == "stage3_logistic_regression"
               and choice["model_path"] == "models/matcher_pipeline.joblib",
               "Stage 5 requires the selected Stage 3 pipeline; refusing a different model")
    pc.require(tuple(choice["feature_schema"]) == tuple(names) == pf.FEATURE_NAMES,
               "Selected model metadata and ordered feature schema disagree")
    threshold = choice["threshold"]
    pc.require(isinstance(threshold, (int, float)) and math.isfinite(threshold) and 0 <= threshold <= 1
               and threshold == config["prediction_threshold"], "Selected and saved thresholds disagree")
    pipeline, saved_threshold = tm.load_matcher(root / "models")
    pc.require(saved_threshold == threshold and tuple(pipeline.feature_names_in_) == pf.FEATURE_NAMES,
               "Loaded model metadata or feature order disagrees with selection")
    pc.require(tuple(pipeline.named_steps["scaler"].feature_names_in_) == pf.FEATURE_NAMES
               and pipeline.named_steps["logistic_regression"].n_features_in_ == len(pf.FEATURE_NAMES),
               "Fitted scaler/classifier feature schema disagrees")
    return pipeline, float(threshold), choice


def build_duplicate_graph(records, pairs, scores, threshold):
    """Observed-only graph; never accepts labels or hierarchy relationships."""
    pc.validate_columns(list(records.values()), pc.CLEANED_FIELDS, "Duplicate graph records")
    pc.require(all(record_id and row["record_id"] == record_id for record_id, row in records.items()),
               "Graph record lookup IDs disagree")
    pc.validate_candidate_pairs(pairs, set(records))
    scores = np.asarray(scores, dtype=float)
    pc.require(scores.shape == (len(pairs),) and np.isfinite(scores).all()
               and ((scores >= 0) & (scores <= 1)).all(), "Invalid candidate model scores")
    pc.require(math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid saved threshold")
    graph = nx.Graph(prediction_threshold=float(threshold), edge_semantics="direct_model_match")
    graph.add_nodes_from(sorted(records))  # Includes no-candidate and no-match records.
    for pair, score in zip(pairs, scores):
        if score >= threshold:
            graph.add_edge(pair["record_id_left"], pair["record_id_right"], model_score=float(score),
                           blocking_rules=pair["blocking_rules"], edge_type="direct_model_match")
    return graph


def deterministic_cluster_id(members):
    encoded = json.dumps(sorted(members), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "clu_" + hashlib.sha256(encoded).hexdigest()


def cluster_memberships(graph):
    pc.require(not graph.is_directed(), "Duplicate components require an undirected graph")
    components = sorted(tuple(sorted(component)) for component in nx.connected_components(graph))
    clusters = {deterministic_cluster_id(members): members for members in components}
    membership = {record_id: cluster_id for cluster_id, members in clusters.items() for record_id in members}
    return membership, clusters


def registration_review_flags(clusters, records):
    """Flag observed conflicts without editing the graph or any membership."""
    flags = []
    for cluster_id, members in sorted(clusters.items()):
        countries = defaultdict(lambda: defaultdict(list))
        for record_id in members:
            country = records[record_id]["country_normalized"].strip()
            registration = records[record_id]["registration_id_normalized"].strip()
            if country and registration:
                countries[country][registration].append(record_id)
        for country, registrations in sorted(countries.items()):
            if len(registrations) > 1:
                flags.append({"cluster_id": cluster_id, "country_normalized": country,
                              "conflicting_registration_ids": json.dumps(sorted(registrations)),
                              "supporting_record_ids": json.dumps(sorted(record_id for ids in registrations.values() for record_id in ids)),
                              "review_reason": "different_nonempty_registration_ids_in_same_nonempty_country",
                              "membership_changed": False})
    return flags


def cluster_summaries(graph, clusters, flags):
    flagged = {row["cluster_id"] for row in flags}
    rows = []
    for cluster_id, members in sorted(clusters.items()):
        direct = graph.subgraph(members).number_of_edges()
        implied = math.comb(len(members), 2)
        rows.append({"cluster_id": cluster_id, "record_count": len(members), "direct_edge_count": direct,
                     "implied_pair_count": implied, "inferred_without_direct_edge_count": implied - direct,
                     "review_required": cluster_id in flagged})
    return rows


def evaluate_clusters(graph, truth):
    """All unordered within-split pairs, not just candidates or direct edges.

    Contingency counts equal exhaustive pair enumeration without materialising
    every pair. Truth is read only after components have already been fixed.
    """
    pc.require(set(graph) == set(truth) and all(truth.values()), "Cluster evaluation truth coverage mismatch")
    membership, clusters = cluster_memberships(graph)
    entity_totals = Counter(truth.values())
    predicted_pairs = sum(math.comb(len(members), 2) for members in clusters.values())
    true_pairs = sum(math.comb(count, 2) for count in entity_totals.values())
    tp = sum(math.comb(count, 2) for members in clusters.values()
             for count in Counter(truth[record_id] for record_id in members).values())
    fp, fn = predicted_pairs - tp, true_pairs - tp
    possible = math.comb(len(graph), 2)
    entity_clusters = defaultdict(set)
    for record_id, cluster_id in membership.items():
        entity_clusters[truth[record_id]].add(cluster_id)
    overmerged = [{"cluster_id": cluster_id, "record_count": len(members),
                   "true_entity_count": len({truth[record_id] for record_id in members}),
                   "true_entity_ids": sorted({truth[record_id] for record_id in members})}
                  for cluster_id, members in sorted(clusters.items())
                  if len({truth[record_id] for record_id in members}) > 1]
    fragmented = [{"true_entity_id": entity, "predicted_cluster_count": len(ids), "cluster_ids": sorted(ids)}
                  for entity, ids in sorted(entity_clusters.items()) if len(ids) > 1]
    def divide(a, b):
        return a / b if b else None
    result = {
        "pairwise_scope": "ALL unordered record pairs within this split, including non-candidates",
        "records": len(graph), "true_entities": len(entity_totals), "cluster_count": len(clusters),
        "singleton_count": sum(len(members) == 1 for members in clusters.values()),
        "largest_cluster_size": max((len(members) for members in clusters.values()), default=0),
        "all_possible_pairs": possible, "predicted_same_cluster_pairs": predicted_pairs,
        "all_true_duplicate_pairs": true_pairs, "tp": tp, "fp": fp, "fn": fn, "tn": possible - tp - fp - fn,
        "pairwise_precision": divide(tp, predicted_pairs), "pairwise_recall": divide(tp, true_pairs),
        "pairwise_f1": divide(2 * tp, 2 * tp + fp + fn),
        "overmerged_cluster_count": len(overmerged), "overmerged_clusters": overmerged,
        "fragmented_true_entity_count": len(fragmented), "fragmented_true_entities": fragmented,
    }
    result["undefined_metrics"] = [name for name in ("pairwise_precision", "pairwise_recall", "pairwise_f1")
                                   if result[name] is None]
    return result


def save_prediction_outputs(directory, graph, records, membership, clusters, flags, summaries):
    rows = [{"cluster_id": membership[record_id], "cluster_size": len(clusters[membership[record_id]]),
             **records[record_id]} for record_id in sorted(graph)]
    edges = [{"cluster_id": membership[left], "record_id_left": left, "record_id_right": right,
              **graph.edges[left, right]} for left, right in sorted(tuple(sorted(edge)) for edge in graph.edges)]
    pc.write_csv(directory / "cluster_membership.csv", MEMBERSHIP_FIELDS, rows)
    pc.write_csv(directory / "direct_edges.csv", EDGE_FIELDS, edges)
    pc.write_csv(directory / "review_flags.csv", FLAG_FIELDS, flags)
    pc.write_csv(directory / "cluster_summary.csv", SUMMARY_FIELDS, summaries)


def run_clustering(root=PROJECT_ROOT):
    from cluster_views import render_duplicate_example, render_reference_hierarchy

    protected_count = verify_preserved_inputs(root)
    pipeline, threshold, choice = load_selected_model(root)
    inputs = tm.load_stage3_inputs(root)
    records = inputs["records"]
    feature_records = {record_id: {field: row[field] for field in pf.NORMALIZED_FIELDS}
                       for record_id, row in records.items()}
    predictions = {}
    for split in tm.EVALUATED_SPLITS:
        split_records = {record_id: records[record_id] for record_id in inputs["split_ids"][split]}
        pairs = inputs["pairs"][split]
        features = pf.build_feature_frame(pairs, feature_records)
        scores, _ = tm.predict_logistic(pipeline, features, threshold)
        graph = build_duplicate_graph(split_records, pairs, scores, threshold)
        membership, clusters = cluster_memberships(graph)
        flags = registration_review_flags(clusters, split_records)
        summaries = cluster_summaries(graph, clusters, flags)
        save_prediction_outputs(root / f"outputs/duplicate_clusters/{split}", graph, split_records,
                                membership, clusters, flags, summaries)
        predictions[split] = {"graph": graph, "membership": membership, "clusters": clusters,
                              "flags": flags, "summaries": summaries, "records": split_records}

    # Components and review flags for BOTH splits exist before this truth-label read.
    truth = tm.load_selected_truth(root / "data/ground_truth/record_entity_map.csv", set(records))
    evaluation = {}
    for split, result in predictions.items():
        evaluation[split] = evaluate_clusters(result["graph"], {
            record_id: truth[record_id] for record_id in result["records"]
        })
        evaluation[split].update({
            "direct_predicted_edges": result["graph"].number_of_edges(),
            "inferred_without_direct_edge_pairs": sum(row["inferred_without_direct_edge_count"] for row in result["summaries"]),
            "review_flag_count": len(result["flags"]),
            "clusters_requiring_review": len({row["cluster_id"] for row in result["flags"]}),
        })
    example = render_duplicate_example(predictions["validation"]["graph"], predictions["validation"]["records"],
                                       root / "outputs/views/validation_duplicate_cluster.html")
    # The reference view receives file paths only, with no predicted clusters or truth mappings.
    reference = render_reference_hierarchy(root / "data/reference/parent_relationships.csv",
                                           root / "data/ground_truth/entities.csv",
                                           root / "outputs/views/reference_corporate_hierarchy.html")
    report = {
        "data_classification": "SYNTHETIC", "stage": 5, "selected_model": choice["chosen_model"],
        "model_path": choice["model_path"], "threshold": threshold, "feature_schema": list(pf.FEATURE_NAMES),
        "clustering_method": "NetworkX connected components on undirected direct match edges",
        "transitivity": "A-B and B-C direct matches imply shared A-C cluster membership, not a direct A-C prediction.",
        "review_policy": "Observed registration conflicts flag review only; never change cluster membership.",
        "approval_status": "Predicted clusters are not approved record merges",
        "metrics_scope": "Cluster pairwise metrics across all record pairs; separate from classifier metrics and blocking recall",
        "evaluation": evaluation, "validation_example": example, "supplied_reference_hierarchy": reference,
        "previous_metrics_unchanged": ["reports/matching_metrics.json", "reports/blocking_report.json"],
        "test_performance": "Hidden; no test records clustered or evaluated",
        "undefined_metric_policy": "null for zero denominators; F1=2TP/(2TP+FP+FN)",
        "preserved_file_count": protected_count,
        "package_versions": {name: version(name) for name in ("networkx", "plotly", "scikit-learn", "joblib")},
    }
    pc.write_json(root / "reports/clustering/cluster_evaluation.json", report)
    verify_preserved_inputs(root)
    print("SYNTHETIC DATA ONLY - Stage 5 predicted duplicate components; no approved merges")
    for split, metrics in evaluation.items():
        print(f"{split}: {metrics['records']} records, {metrics['cluster_count']} clusters, "
              f"{metrics['singleton_count']} singletons, largest {metrics['largest_cluster_size']}, "
              f"{metrics['direct_predicted_edges']} direct edges")
        print(f"  ALL-pairs precision={metrics['pairwise_precision']}, recall={metrics['pairwise_recall']}, "
              f"F1={metrics['pairwise_f1']}; TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}; "
              f"overmerged={metrics['overmerged_cluster_count']}, fragmented entities={metrics['fragmented_true_entity_count']}, "
              f"review clusters={metrics['clusters_requiring_review']}")
    print("Saved outputs/duplicate_clusters/, outputs/views/, and reports/clustering/cluster_evaluation.json")
    print("PASS: preserved prior artifacts, selected metadata/schema, no retraining; test results hidden")
    return report


if __name__ == "__main__":
    run_clustering()
