"""Read saved artifacts without invoking any pipeline entry point.

The Stage 6 evaluator's inventory guard predates the UI files. This adapter checks
the hashes of existing runtime artifacts, permitting additive UI source files and
an environment without the optional embedding packages. It never edits a freeze.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import build_clusters as bc
import cluster_views as cv
import pair_features as pf
import prepare_candidates as pc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = "reports/test_evaluation/v1"
ORIGINAL_REPORTS = f"{FREEZE_DIR}/original"
ORIGINAL_OUTPUTS = "outputs/test_evaluation/v1/original"
SPLIT_DIRECTORIES = {
    "train": "outputs/duplicate_clusters/train",
    "validation": "outputs/duplicate_clusters/validation",
    "test": f"{ORIGINAL_OUTPUTS}/clusters",
}
CLUSTER_SCHEMAS = {
    "cluster_membership.csv": bc.MEMBERSHIP_FIELDS, "direct_edges.csv": bc.EDGE_FIELDS,
    "review_flags.csv": bc.FLAG_FIELDS, "cluster_summary.csv": bc.SUMMARY_FIELDS,
}
RUNTIME_PATHS = (
    f"{FREEZE_DIR}/freeze_manifest.json", f"{FREEZE_DIR}/freeze_manifest.sha256",
    f"{ORIGINAL_REPORTS}/artifacts.json", f"{ORIGINAL_REPORTS}/evaluation.json",
    "models/next_stage_choice.json", "models/feature_names.json", "models/matcher_config.json",
    "models/package_versions.json", "models/matcher_pipeline.joblib",
    "reports/embedding_experiment/comparison.json", "data/raw/client_records.csv",
    "data/reference/parent_relationships.csv", "data/ground_truth/entities.csv",
    "data/ground_truth/record_entity_map.csv",
    "build_clusters.py", "cluster_views.py", "pair_features.py", "prepare_candidates.py", "train_matcher.py",
    *(f"{directory}/{name}" for directory in SPLIT_DIRECTORIES.values() for name in CLUSTER_SCHEMAS),
)


class ArtifactError(RuntimeError):
    """A readable local-project problem, shown in the app without a traceback."""


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def project_path(root, relative):
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in relative,
            "An artifact contains an invalid relative path.")
    path = root.joinpath(*pure.parts)
    require(path.resolve().is_relative_to(root.resolve()), "An artifact path leaves the project folder.")
    return path


def digest(path):
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as error:
        raise ArtifactError(f"Cannot read required artifact: {path.name}. Restore the saved project files.") from error


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Cannot read JSON artifact: {path.name}. Restore the original saved file.") from error


def artifact_stamp(root):
    """Invalidate Streamlit caches if a required file is removed or changed."""
    entries = []
    for relative in RUNTIME_PATHS:
        path = project_path(root, relative)
        require(path.is_file(), f"Missing required artifact: {relative}. Restore it from the completed project.")
        stat = path.stat()
        entries.append((relative, stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


@dataclass
class Catalog:
    root: Path
    freeze: dict
    report: dict
    choice: dict
    expected: dict
    freeze_sha256: str

    def checked_path(self, relative):
        path = project_path(self.root, relative)
        require(relative in self.expected, f"Artifact is outside the saved integrity manifests: {relative}")
        require(digest(path) == self.expected[relative], f"Saved artifact changed: {relative}. Restore the original file.")
        return path


def load_catalog(root):
    """Validate the original report and frozen inputs; do not recompute metrics."""
    root = Path(root)
    artifact_stamp(root)
    path = root / FREEZE_DIR / "freeze_manifest.json"
    freeze_hash = digest(path)
    require((root / FREEZE_DIR / "freeze_manifest.sha256").read_text(encoding="utf-8").strip() == freeze_hash,
            "The original v1 freeze manifest checksum disagrees. Restore the saved manifest.")
    freeze = read_json(path)
    commits = read_json(root / ORIGINAL_REPORTS / "artifacts.json")
    try:
        require(freeze["evaluation_version"] == "v1" and commits["freeze_sha256"] == freeze_hash,
                "The original test artifacts do not belong to the v1 freeze.")
        expected = dict(freeze["protected_sha256"])
        expected.update({f"{ORIGINAL_REPORTS}/{name}": value for name, value in commits["report_sha256"].items()})
        expected.update({f"{ORIGINAL_OUTPUTS}/{name}": value for name, value in commits["output_sha256"].items()})
        catalog = Catalog(root, freeze, {}, {}, expected, freeze_hash)
        uncommitted = {f"{FREEZE_DIR}/freeze_manifest.json", f"{FREEZE_DIR}/freeze_manifest.sha256",
                       f"{ORIGINAL_REPORTS}/artifacts.json"}
        for relative in RUNTIME_PATHS:
            if relative not in uncommitted:
                catalog.checked_path(relative)
        catalog.report = read_json(root / ORIGINAL_REPORTS / "evaluation.json")
        catalog.choice = read_json(root / "models/next_stage_choice.json")
        report, choice = catalog.report, catalog.choice
        require(report["data_classification"] == "SYNTHETIC" and report["run_kind"] == "original held-out evaluation"
                and report["freeze_sha256"] == freeze_hash, "Expected the original frozen synthetic test report.")
        require(report["feature_schema"] == choice["feature_schema"] == freeze["feature_schema"] == list(pf.FEATURE_NAMES),
                "Selected model and frozen feature schemas disagree.")
        require(report["threshold"] == choice["threshold"] == freeze["threshold"],
                "Selected model and frozen thresholds disagree.")
        require(choice == freeze["selected_model"] and choice["chosen_model"] == "stage3_logistic_regression",
                "The saved selection is not the frozen Stage 3 model.")
        require(set(report["metrics"]) == {"blocking", "candidate_classification", "end_to_end_direct_pair", "clustering"},
                "Unexpected original test report schema.")
        # Check report shapes here so missing keys become a readable artifact error.
        block, cluster = report["metrics"]["blocking"], report["metrics"]["clustering"]
        for key in ("record_count", "entity_count", "candidate_pairs", "all_true_duplicate_pairs", "total_possible_pairs"):
            require(isinstance(block[key], int) and block[key] >= 0, "Invalid saved blocking counts.")
        for method in ("rule_based", "logistic_regression"):
            row = report["metrics"]["candidate_classification"][method]
            for key in ("precision", "recall", "f1", "tp", "fp", "tn", "fn"):
                _ = row[key]
            _ = report["metrics"]["end_to_end_direct_pair"][method]["end_to_end_direct_pair_recall"]
        _ = cluster["pairwise_f1"]
        return catalog
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("A saved report or model metadata has an unexpected schema. Restore the original artifacts.") from error


def read_verified_csv(catalog, relative, fields):
    try:
        return pc.read_csv(catalog.checked_path(relative), tuple(fields))
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Cannot load saved CSV: {relative}. Check its original schema and contents.") from error


def load_samples(catalog):
    rows = read_verified_csv(catalog, "data/raw/client_records.csv", pc.OBSERVED_FIELDS)
    try:
        pc.validate_record_ids(rows, "Synthetic samples")
    except ValueError as error:
        raise ArtifactError(str(error)) from error
    return {row["record_id"]: row for row in rows}


def load_cluster_bundle(catalog, split):
    require(split in SPLIT_DIRECTORIES, "Select train, validation or test saved clusters.")
    directory = SPLIT_DIRECTORIES[split]
    data = {name: read_verified_csv(catalog, f"{directory}/{name}", fields)
            for name, fields in CLUSTER_SCHEMAS.items()}
    members = data["cluster_membership.csv"]
    summaries = data["cluster_summary.csv"]
    try:
        pc.validate_record_ids(members, "Saved membership")
        by_cluster = {}
        for row in members:
            by_cluster.setdefault(row["cluster_id"], []).append(row)
        require(set(by_cluster) == {row["cluster_id"] for row in summaries}, "Cluster membership and summaries disagree.")
        for row in summaries:
            require(int(row["record_count"]) == len(by_cluster[row["cluster_id"]]), "Saved cluster sizes disagree.")
        for row in data["direct_edges.csv"]:
            ids = {member["record_id"] for member in by_cluster[row["cluster_id"]]}
            require({row["record_id_left"], row["record_id_right"]} <= ids, "Saved edge leaves its cluster.")
        return {"members": members, "by_cluster": by_cluster, "summaries": summaries,
                "edges": data["direct_edges.csv"], "flags": data["review_flags.csv"]}
    except (ValueError, KeyError) as error:
        raise ArtifactError("Saved cluster outputs disagree. Restore their original files.") from error


def load_reference_hierarchy(catalog):
    """Only supplied relationships create nodes/edges; canonical names label them."""
    rows = read_verified_csv(catalog, "data/reference/parent_relationships.csv", pc.PARENT_FIELDS)
    try:
        graph = cv.build_reference_graph(rows)
        names = cv.read_canonical_display_names(catalog.checked_path("data/ground_truth/entities.csv"), set(graph))
    except (ValueError, OSError, KeyError) as error:
        raise ArtifactError("The supplied reference hierarchy is missing valid relationships or display names.") from error
    return graph, names, rows


def load_evaluation_truth(catalog):
    """Evaluation display only. Never called by live scoring or graph construction."""
    rows = read_verified_csv(catalog, "data/ground_truth/record_entity_map.csv", pc.TRUTH_FIELDS)
    return {row["record_id"]: row["true_entity_id"] for row in rows}


def load_frozen_model(catalog):
    # Only load the trusted, hash-checked local joblib artifact. No fit call exists.
    for relative in ("models/matcher_pipeline.joblib", "models/feature_names.json", "models/matcher_config.json",
                     "models/next_stage_choice.json", "models/package_versions.json"):
        catalog.checked_path(relative)
    try:
        pipeline, threshold, choice = bc.load_selected_model(catalog.root)
        require(choice == catalog.choice and threshold == catalog.freeze["threshold"], "Loaded model differs from freeze.")
        return pipeline, threshold
    except (ValueError, OSError, KeyError, AttributeError, ImportError) as error:
        raise ArtifactError("Cannot load the selected frozen model. Use Python 3.12 and the pinned app requirements.") from error
