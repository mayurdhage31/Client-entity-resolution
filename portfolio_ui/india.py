"""Read-only Indian demo adapter. Teaching labels never determine predictions."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx

import prepare_candidates as pc
from build_india_demo import COMPANY_FIELDS, DATA_DIR, OUTPUT_DIR, RELATIONSHIP_FIELDS
from portfolio_ui import artifacts as ad


REQUIRED_PATHS = (
    f"{DATA_DIR}/observed/client_records.csv", f"{DATA_DIR}/reference/companies.csv",
    f"{DATA_DIR}/reference/public_relationships.csv", f"{DATA_DIR}/ground_truth/record_entity_map.csv",
    f"{DATA_DIR}/education/guided_examples.json", f"{DATA_DIR}/dataset_info.json", f"{OUTPUT_DIR}/prediction_run.json",
    *(f"{OUTPUT_DIR}/clusters/{name}" for name in ad.CLUSTER_SCHEMAS),
)


def demo_stamp(root):
    stamp = []
    for relative in REQUIRED_PATHS:
        path = ad.project_path(root, relative)
        ad.require(path.is_file(), f"Missing Indian demo artifact: {relative}. Restore the separate demo dataset and saved predictions.")
        stat = path.stat()
        stamp.append((relative, stat.st_size, stat.st_mtime_ns))
    return tuple(stamp)


@dataclass
class Demo:
    root: Path
    info: dict
    run: dict
    records: dict
    bundle: dict
    companies: list
    relationships: list
    examples: list


def checked_data_path(demo, relative):
    path = ad.project_path(demo.root, relative)
    ad.require(relative in demo.run["data_sha256"] and ad.digest(path) == demo.run["data_sha256"][relative],
               "An Indian demo data file changed. Restore its saved version.")
    return path


def load_demo(root, catalog):
    root = Path(root)
    demo_stamp(root)
    run = ad.read_json(root / OUTPUT_DIR / "prediction_run.json")
    try:
        ad.require(run["dataset_id"] == "india_demo_v1" and run["model_sha256"] == catalog.freeze["selected_model_sha256"]
                   and run["original_freeze_sha256"] == catalog.freeze_sha256, "Indian predictions do not use the selected original frozen model.")
        ad.require(run["feature_schema"] == catalog.choice["feature_schema"] and run["threshold"] == catalog.choice["threshold"]
                   and run["blocking_configuration"] == catalog.freeze["blocking"], "Indian prediction settings disagree with the frozen pipeline.")
        ad.require(not run["truth_used_for_predictions"] and not run["reference_relationships_used_for_predictions"],
                   "Indian demo prediction provenance is invalid.")
        for relative, expected in {**run["data_sha256"], **run["prediction_files_sha256"]}.items():
            path = ad.project_path(root, relative)
            ad.require(ad.digest(path) == expected, f"Indian demo artifact changed: {relative}. Restore its saved version.")
        records = pc.read_csv(root / DATA_DIR / "observed/client_records.csv", pc.OBSERVED_FIELDS)
        pc.validate_record_ids(records, "Indian demo observations")
        by_id = {row["record_id"]: row for row in records}
        members = pc.read_csv(root / OUTPUT_DIR / "clusters/cluster_membership.csv", ad.bc.MEMBERSHIP_FIELDS)
        pc.validate_record_ids(members, "Indian demo membership")
        ad.require({row["record_id"] for row in members} == set(by_id), "Indian groups must contain every Indian record and no v1 records.")
        for row in members:
            ad.require({field: row[field] for field in pc.OBSERVED_FIELDS} == by_id[row["record_id"]],
                       "Indian group members disagree with the actual observed Indian fields.")
        by_cluster = {}
        for row in members:
            by_cluster.setdefault(row["cluster_id"], []).append(row)
        tables = {name: pc.read_csv(root / OUTPUT_DIR / "clusters" / name, fields)
                  for name, fields in ad.CLUSTER_SCHEMAS.items() if name != "cluster_membership.csv"}
        summaries, edges, flags = tables["cluster_summary.csv"], tables["direct_edges.csv"], tables["review_flags.csv"]
        ad.require({row["cluster_id"] for row in summaries} == set(by_cluster), "Indian group summaries disagree with memberships.")
        for row in summaries:
            ad.require(int(row["record_count"]) == len(by_cluster[row["cluster_id"]]), "Indian group member counts disagree.")
        for row in edges:
            allowed = {member["record_id"] for member in by_cluster[row["cluster_id"]]}
            ad.require({row["record_id_left"], row["record_id_right"]} <= allowed, "A saved Indian edge leaves its group.")
        companies = pc.read_csv(root / DATA_DIR / "reference/companies.csv", COMPANY_FIELDS)
        relationships = pc.read_csv(root / DATA_DIR / "reference/public_relationships.csv", RELATIONSHIP_FIELDS)
        examples = ad.read_json(root / DATA_DIR / "education/guided_examples.json")["examples"]
        for example in examples:
            ad.require(example["record_id_left"] != example["record_id_right"] and
                       {example["record_id_left"], example["record_id_right"]} <= set(by_id), "A guided example references a record outside the Indian demo.")
        return Demo(root, ad.read_json(root / DATA_DIR / "dataset_info.json"), run, by_id,
                    {"members": members, "by_cluster": by_cluster, "summaries": summaries, "edges": edges, "flags": flags},
                    companies, relationships, examples)
    except (KeyError, ValueError, OSError, TypeError) as error:
        raise ad.ArtifactError("Indian demo files have an unexpected schema. Restore the saved demo data and outputs.") from error


def record_label(record):
    return f"{record['company_name'] or '(company name missing)'} | {record['source_system']} | {record['record_id'][-8:]}"


def record_options(demo, exclude=None):
    return sorted((record_id for record_id in demo.records if record_id != exclude),
                  key=lambda record_id: (demo.records[record_id]["company_name"].casefold(), demo.records[record_id]["source_system"], record_id))


def educational_expectation(demo, example):
    """Read separate simulation labels only to explain the expected teaching case."""
    path = checked_data_path(demo, f"{DATA_DIR}/ground_truth/record_entity_map.csv")
    truth = {row["record_id"]: row["true_entity_id"] for row in pc.read_csv(path, pc.TRUTH_FIELDS)}
    same = truth[example["record_id_left"]] == truth[example["record_id_right"]]
    expected = "same legal entity" if same else "different legal entities"
    ad.require(expected == example["expected_outcome"], "A teaching label disagrees with the separate simulation truth.")
    return expected


def filtered_groups(demo, company=None, source=None, size_filter="All groups", review_only=False):
    flagged = {row["cluster_id"] for row in demo.bundle["flags"]}
    selected = []
    for group, members in demo.bundle["by_cluster"].items():
        if size_filter == "Single-record groups" and len(members) != 1:
            continue
        if size_filter == "Multiple-record groups" and len(members) <= 1:
            continue
        if review_only and group not in flagged:
            continue
        if any((company is None or row["company_name"] == company) and (source is None or row["source_system"] == source) for row in members):
            selected.append(group)
    return sorted(selected, key=lambda group: (-len(demo.bundle["by_cluster"][group]), group))


def group_label(demo, group):
    members = demo.bundle["by_cluster"][group]
    names = sorted({row["company_name"] for row in members if row["company_name"]}, key=lambda name: (-len(name), name))
    name = names[0] if names else "Company name missing"
    flag = " | Review flag" if any(row["cluster_id"] == group for row in demo.bundle["flags"]) else ""
    return f"{name} | {len(members)} record(s){flag}"


def reference_graph(demo):
    """Only verified public reference rows, never duplicate group membership."""
    names = {row["reference_entity_id"]: row["company_name"] for row in demo.companies}
    graph = nx.DiGraph()
    for row in demo.relationships:
        parent, child = row["parent_entity_id"], row["child_entity_id"]
        ad.require(parent in names and child in names and parent != child, "Invalid public reference relationship.")
        url = urlparse(row["source_url"])
        ad.require(url.scheme == "https" and url.hostname == "www.aboutbajajfinserv.com", "Unexpected public relationship source URL.")
        graph.add_edge(parent, child, **{key: value for key, value in row.items() if key not in ("parent_entity_id", "child_entity_id")})
    ad.require(nx.is_directed_acyclic_graph(graph), "Public reference relationships contain a cycle.")
    return graph, names
