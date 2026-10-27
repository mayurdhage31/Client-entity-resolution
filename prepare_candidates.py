"""Stage 2: clean SYNTHETIC records, persist family splits, and block candidates.

Truth and hierarchy are used for partitioning and evaluation only. The blocking
function accepts only observed fields and their deterministic normalisations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Sequence

import networkx as nx
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data/raw/client_records.csv"
TRUTH_PATH = PROJECT_ROOT / "data/ground_truth/record_entity_map.csv"
PARENTS_PATH = PROJECT_ROOT / "data/reference/parent_relationships.csv"
CLEANED_PATH = PROJECT_ROOT / "data/processed/cleaned_records.csv"
SPLITS_PATH = PROJECT_ROOT / "data/splits/record_splits.csv"
REPORT_PATH = PROJECT_ROOT / "reports/blocking_report.json"

SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.60, 0.20, 0.20)
SPLIT_STRATEGY = "balanced_family_groups_v1"
DETAIL_FIELDS = ("company_name", "registration_id", "address", "city", "postcode", "country")
OBSERVED_FIELDS = ("record_id", "source_system", *DETAIL_FIELDS)
NORMALIZED_FIELDS = tuple(f"{field}_normalized" for field in DETAIL_FIELDS)
CLEANED_FIELDS = (*OBSERVED_FIELDS, *NORMALIZED_FIELDS)
TRUTH_FIELDS = ("record_id", "true_entity_id")
PARENT_FIELDS = ("child_entity_id", "parent_entity_id", "relationship_source")
SPLIT_FIELDS = (
    "record_id", "true_entity_id", "split_group_id", "split", "split_seed", "split_ratios",
    "split_strategy",
)
PAIR_FIELDS = ("record_id_left", "record_id_right", "blocking_rules")
RULE_NAMES = (
    "country_registration",
    "country_postcode_name_prefix",
    "country_city_name_token",
)
FALLBACK_RULE_NAMES = ("fuzzy_name_top_k", "fuzzy_address_top_k")
ALL_METHOD_NAMES = (*RULE_NAMES, *FALLBACK_RULE_NAMES)


@dataclass(frozen=True)
class BlockingConfig:
    """Static, reviewable rules; no vocabulary or thresholds learned from test data."""

    name_prefix_length: int = 4
    minimum_token_length: int = 4
    excluded_name_tokens: tuple[str, ...] = (
        "and", "the", "co", "company", "corp", "corporation", "inc", "incorporated",
        "limited", "ltd", "llc", "llp", "plc", "public", "liability", "partnership",
        "advisory", "capital", "consulting", "group", "holding", "holdings",
        "industries", "international", "global", "logistics", "partners", "regional",
        "resources", "service", "services", "solutions", "technology", "trading",
    )

    def __post_init__(self) -> None:
        if self.name_prefix_length < 1 or self.minimum_token_length < 1:
            raise ValueError("Prefix and minimum token lengths must be positive")


DEFAULT_BLOCKING_CONFIG = BlockingConfig()


@dataclass(frozen=True)
class FallbackConfig:
    """Initial retrieval settings, never final match/classification thresholds."""

    enabled: bool = True
    company_name_top_k: int = 5
    company_name_min_score: float = 80.0
    address_top_k: int = 3
    address_min_score: float = 85.0

    def __post_init__(self) -> None:
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 1
               for k in (self.company_name_top_k, self.address_top_k)):
            raise ValueError("Fallback neighbour counts must be positive integers")
        if any(not 0 <= score <= 100
               for score in (self.company_name_min_score, self.address_min_score)):
            raise ValueError("WRatio cutoffs must be between 0 and 100")


DEFAULT_FALLBACK_CONFIG = FallbackConfig()
Row = dict[str, str]


def require(condition: bool, message: str) -> None:
    """Use real exceptions so validation also runs with python -O."""
    if not condition:
        raise ValueError(message)


def validate_columns(rows: Sequence[Row], fields: tuple[str, ...], label: str) -> None:
    for row in rows:
        require(set(row) == set(fields), f"{label}: unexpected or missing columns")
        require(all(isinstance(value, str) for value in row.values()),
                f"{label}: all cells must be strings; use empty strings for missing cells")


def validate_record_ids(rows: Sequence[Row], label: str) -> None:
    ids = [row["record_id"] for row in rows]
    require(all(value and value == value.strip() for value in ids),
            f"{label}: record IDs must be populated and have no surrounding whitespace")
    require(len(ids) == len(set(ids)), f"{label}: duplicate record IDs")


def read_csv(path: Path, fields: tuple[str, ...]) -> list[Row]:
    """csv reads text without numeric/NA inference: 00123 and NA stay strings."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        require(len(header) == len(set(header)), f"{path.name}: duplicate column names")
        missing = set(fields) - set(header)
        require(not missing, f"{path.name}: missing required columns: {sorted(missing)}")
        require(set(header) == set(fields), f"{path.name}: unexpected columns")
        rows = list(reader)
    # A missing CSV cell is '', whereas malformed short/long rows produce None.
    validate_columns(rows, fields, path.name)
    return rows


def write_csv(path: Path, fields: tuple[str, ...], rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_observed_records(path: Path) -> list[Row]:
    rows = read_csv(path, OBSERVED_FIELDS)
    validate_record_ids(rows, "Observed records")
    require(bool(rows), "Observed records: input is empty")
    require(all(row["source_system"].strip() for row in rows), "Missing source_system")
    return rows


def normalize_text(value: str | None) -> str:
    """NFKC, Unicode case folding, punctuation/symbol removal, and tidy spacing.

    Accents are retained. Only None, empty, or whitespace-only input is missing;
    literal identifiers such as NA are not automatically converted to missing.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("Normalisation expects text, preserving any leading zeros")
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    return " ".join("".join(char if char.isalnum() else " " for char in value).split())


def normalize_company_name(value: str | None) -> str:
    aliases = {"limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co"}
    return " ".join(aliases.get(token, token) for token in normalize_text(value).split())


def normalize_compact(value: str | None) -> str:
    """Compact identifiers/postcodes without converting them to numbers."""
    return "".join(normalize_text(value).split())


def normalize_address(value: str | None) -> str:
    aliases = {
        "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
        "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
        "ter": "terrace", "apt": "flat", "apartment": "flat", "fl": "floor",
    }
    return " ".join(aliases.get(token, token) for token in normalize_text(value).split())


def normalize_country(value: str | None) -> str:
    country = normalize_text(value)
    if country.replace(" ", "") in {"uk", "gb", "gbr", "unitedkingdom", "greatbritain"}:
        return "gb"
    return country


def clean_records(records: Sequence[Row]) -> list[Row]:
    """Retain every original cell and append six observed-only normalised fields."""
    validate_columns(records, OBSERVED_FIELDS, "Observed records")
    validate_record_ids(records, "Observed records")
    normalizers = {
        "company_name": normalize_company_name,
        "registration_id": normalize_compact,
        "address": normalize_address,
        "city": normalize_text,
        "postcode": normalize_compact,
        "country": normalize_country,
    }
    return [
        {**record, **{f"{field}_normalized": function(record[field])
                      for field, function in normalizers.items()}}
        for record in records
    ]


def build_split_groups(truth: Sequence[Row], parents: Sequence[Row]) -> dict[str, str]:
    """Use truth IDs and undirected corporate components ONLY for partitioning."""
    validate_columns(truth, TRUTH_FIELDS, "Truth map")
    validate_record_ids(truth, "Truth map")
    require(all(row["true_entity_id"].strip() for row in truth), "Missing truth entity ID")
    validate_columns(parents, PARENT_FIELDS, "Parent relationships")
    entity_ids = {row["true_entity_id"] for row in truth}
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(entity_ids))  # Includes independent entities.
    edges = [(row["parent_entity_id"], row["child_entity_id"]) for row in parents]
    require(all(parent in entity_ids and child in entity_ids for parent, child in edges),
            "Invalid parent relationship reference")
    require(all(parent != child for parent, child in edges), "Self-parenting relationship")
    require(len(edges) == len(set(edges)), "Duplicate parent relationships")
    require(all(row["relationship_source"].strip() for row in parents),
            "Missing relationship_source")
    graph.add_edges_from(edges)
    require(nx.is_directed_acyclic_graph(graph), "Corporate hierarchy has a cycle")
    groups: dict[str, str] = {}
    for component in nx.connected_components(graph.to_undirected()):
        # Stable membership hash makes changed family definitions detectable.
        members = json.dumps(sorted(component), ensure_ascii=False).encode("utf-8")
        group_id = "grp_" + hashlib.sha256(members).hexdigest()
        groups.update({entity_id: group_id for entity_id in component})
    return groups


def split_metadata(seed: int, ratios: tuple[float, ...]) -> dict[str, str]:
    require(len(ratios) == 3 and all(value > 0 for value in ratios)
            and math.isclose(sum(ratios), 1.0), "Use three positive split ratios summing to 1")
    return {"split_seed": str(seed), "split_ratios": "/".join(str(value) for value in ratios),
            "split_strategy": SPLIT_STRATEGY}


def create_split_assignments(
    truth: Sequence[Row], groups: dict[str, str], seed: int = SEED,
    ratios: tuple[float, ...] = SPLIT_RATIOS,
) -> list[Row]:
    metadata = split_metadata(seed, ratios)
    group_ids = sorted(set(groups.values()))
    random.Random(seed).shuffle(group_ids)
    record_weights = Counter(groups[row["true_entity_id"]] for row in truth)
    entity_weights = Counter(groups.values())
    # Place large families first; seeded shuffle breaks ties without inspecting
    # names, candidate pairs, or blocking performance. Balance records AND entities.
    group_ids.sort(key=lambda group: (-entity_weights[group], -record_weights[group]))
    loads = {split: [0, 0] for split in SPLIT_NAMES}
    totals = (len(truth), len(groups))
    group_splits: dict[str, str] = {}
    for group in group_ids:
        weight = (record_weights[group], entity_weights[group])
        def allocation_cost(split: str) -> float:
            target = ratios[SPLIT_NAMES.index(split)]
            # Scale by target size so smaller partitions receive their fair share
            # of large families too. No candidate quality enters this objective.
            return sum((((loads[split][i] + weight[i]) / totals[i] - target) ** 2
                        - (loads[split][i] / totals[i] - target) ** 2) / target
                       for i in (0, 1))
        chosen = min(SPLIT_NAMES, key=allocation_cost)
        group_splits[group] = chosen
        loads[chosen] = [loads[chosen][i] + weight[i] for i in (0, 1)]
    return [
        {"record_id": row["record_id"], "true_entity_id": row["true_entity_id"],
         "split_group_id": groups[row["true_entity_id"]],
         "split": group_splits[groups[row["true_entity_id"]]], **metadata}
        for row in sorted(truth, key=lambda row: row["record_id"])
    ]


def assert_split_isolation(
    assignments: Sequence[Row], records: Sequence[Row], truth: Sequence[Row],
    groups: dict[str, str], seed: int = SEED, ratios: tuple[float, ...] = SPLIT_RATIOS,
) -> None:
    validate_columns(assignments, SPLIT_FIELDS, "Saved splits")
    validate_record_ids(assignments, "Saved splits")
    validate_columns(truth, TRUTH_FIELDS, "Truth map")
    validate_record_ids(truth, "Truth map")
    expected_truth = {row["record_id"]: row["true_entity_id"] for row in truth}
    record_ids = {row["record_id"] for row in records}
    require(record_ids == set(expected_truth), "Truth mappings must cover exactly all observed records")
    require({row["record_id"] for row in assignments} == record_ids,
            "Saved splits do not cover exactly the current records")
    entity_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    metadata = split_metadata(seed, ratios)
    for row in assignments:
        entity_id = expected_truth[row["record_id"]]
        require(row["true_entity_id"] == entity_id, "Saved split truth mapping has changed")
        require(row["split_group_id"] == groups[entity_id], "Saved corporate family membership has changed")
        require(row["split"] in SPLIT_NAMES, "Unknown split name")
        require(all(row[key] == value for key, value in metadata.items()),
                "Saved split seed, ratios, or strategy differ from configuration")
        entity_splits[entity_id].add(row["split"])
        family_splits[groups[entity_id]].add(row["split"])
    require(all(len(splits) == 1 for splits in entity_splits.values()), "Entity crosses splits")
    require(all(len(splits) == 1 for splits in family_splits.values()), "Corporate family crosses splits")
    record_sets = [{row["record_id"] for row in assignments if row["split"] == split}
                   for split in SPLIT_NAMES]
    require(all(not left & right for left, right in combinations(record_sets, 2)), "Record crosses splits")


def load_or_create_splits(
    path: Path, records: Sequence[Row], truth: Sequence[Row], groups: dict[str, str],
    seed: int = SEED, ratios: tuple[float, ...] = SPLIT_RATIOS,
) -> tuple[list[Row], bool]:
    reused = path.exists()
    assignments = (read_csv(path, SPLIT_FIELDS) if reused
                   else create_split_assignments(truth, groups, seed, ratios))
    try:
        assert_split_isolation(assignments, records, truth, groups, seed, ratios)
    except ValueError as error:
        raise ValueError(
            f"Split validation failed: {error}. Existing assignments are never silently replaced; "
            "restore compatible inputs or explicitly archive the split file for a new experiment."
        ) from error
    if not reused:
        write_csv(path, SPLIT_FIELDS, assignments)
    return assignments, reused


def informative_name_tokens(name: str, config: BlockingConfig = DEFAULT_BLOCKING_CONFIG) -> tuple[str, ...]:
    excluded = set(config.excluded_name_tokens)
    return tuple(sorted({token for token in name.split()
                         if len(token) >= config.minimum_token_length
                         and any(char.isalpha() for char in token) and token not in excluded}))


def generate_candidate_pairs(
    cleaned_records: Sequence[Row], config: BlockingConfig = DEFAULT_BLOCKING_CONFIG,
) -> list[Row]:
    """Union observed-only blocks. Call separately for each already assigned split.

    IDs only identify/canonicalise pair endpoints; their contents never form a key.
    No source-system filter is used, so same-source pairs are also allowed.
    """
    # Reject unexpected metadata (including truth/family/split columns) at the boundary.
    validate_columns(cleaned_records, CLEANED_FIELDS, "Blocking inputs")
    validate_record_ids(cleaned_records, "Blocking inputs")
    buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for record in cleaned_records:
        country = record["country_normalized"]
        name = record["company_name_normalized"]
        prefix = "".join(char for char in name if char.isalnum())[:config.name_prefix_length]
        keys = [
            (RULE_NAMES[0], country, record["registration_id_normalized"]),
            (RULE_NAMES[1], country, record["postcode_normalized"], prefix),
        ]
        keys.extend((RULE_NAMES[2], country, record["city_normalized"], token)
                    for token in informative_name_tokens(name, config))
        for key in keys:
            # Empty values NEVER share a block, even when both records are missing.
            if all(value.strip() for value in key[1:]):
                buckets[key].append(record["record_id"])

    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key, members in buckets.items():
        for left, right in combinations(sorted(set(members)), 2):
            if left != right:
                reasons[left, right].add(key[0])
    pairs = [{"record_id_left": left, "record_id_right": right,
              "blocking_rules": "|".join(sorted(rules))}
             for (left, right), rules in sorted(reasons.items())]
    validate_candidate_pairs(pairs, {row["record_id"] for row in cleaned_records})
    return pairs


def top_fuzzy_neighbors(
    query_id: str, query_text: str, choices: dict[str, str], top_k: int, cutoff: float,
) -> list[tuple[str, float]]:
    """Exhaustive demo retrieval; exclude self/empty text BEFORE scoring/ranking.

    Use the full score to rank, then the opaque ID only to resolve exact ties.
    Input text is already normalised; RapidFuzz applies no further processor.
    """
    if not query_text.strip():
        return []
    scored = []
    for record_id, text in choices.items():
        if record_id == query_id or not text.strip():
            continue
        score = fuzz.WRatio(query_text, text, processor=None)
        if score >= cutoff:
            scored.append((record_id, score))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def union_candidate_pairs(*pair_sets: Sequence[Row]) -> list[Row]:
    """Combine unordered pairs and retain every exact/fallback retrieval reason."""
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pairs in pair_sets:
        validate_columns(pairs, PAIR_FIELDS, "Candidate union")
        for row in pairs:
            left, right = sorted((row["record_id_left"], row["record_id_right"]))
            if left != right:
                reasons[left, right].update(row["blocking_rules"].split("|"))
    return [{"record_id_left": left, "record_id_right": right,
             "blocking_rules": "|".join(sorted(methods))}
            for (left, right), methods in sorted(reasons.items())]


def generate_fallback_candidates(
    cleaned_records: Sequence[Row], config: FallbackConfig = DEFAULT_FALLBACK_CONFIG,
) -> list[Row]:
    """Search only the supplied split; labels and geographic filters are absent."""
    validate_columns(cleaned_records, CLEANED_FIELDS, "Fallback inputs")
    validate_record_ids(cleaned_records, "Fallback inputs")
    if not config.enabled:
        return []
    names = {row["record_id"]: row["company_name_normalized"] for row in cleaned_records
             if row["company_name_normalized"].strip()}
    addresses = {row["record_id"]: row["address_normalized"] for row in cleaned_records
                 if row["address_normalized"].strip()}
    directed_pairs = []
    for row in sorted(cleaned_records, key=lambda row: row["record_id"]):
        record_id = row["record_id"]
        if record_id in names:
            text, choices = names[record_id], names
            top_k, cutoff = config.company_name_top_k, config.company_name_min_score
            method = FALLBACK_RULE_NAMES[0]
        elif record_id in addresses:
            # Only missing-name QUERIES use addresses. Targets may have a name.
            text, choices = addresses[record_id], addresses
            top_k, cutoff = config.address_top_k, config.address_min_score
            method = FALLBACK_RULE_NAMES[1]
        else:
            continue
        for neighbor_id, _ in top_fuzzy_neighbors(record_id, text, choices, top_k, cutoff):
            directed_pairs.append({"record_id_left": record_id, "record_id_right": neighbor_id,
                                   "blocking_rules": method})
    pairs = union_candidate_pairs(directed_pairs)
    validate_candidate_pairs(pairs, {row["record_id"] for row in cleaned_records})
    return pairs


def generate_revised_candidate_pairs(
    cleaned_records: Sequence[Row], config: BlockingConfig = DEFAULT_BLOCKING_CONFIG,
    fallback_config: FallbackConfig = DEFAULT_FALLBACK_CONFIG,
) -> list[Row]:
    """Pure observed-only union; even already-blocked neighbors consume top-k slots."""
    pairs = union_candidate_pairs(generate_candidate_pairs(cleaned_records, config),
                                  generate_fallback_candidates(cleaned_records, fallback_config))
    validate_candidate_pairs(pairs, {row["record_id"] for row in cleaned_records})
    return pairs


def validate_candidate_pairs(pairs: Sequence[Row], allowed_ids: set[str]) -> None:
    validate_columns(pairs, PAIR_FIELDS, "Candidate pairs")
    endpoints = [(row["record_id_left"], row["record_id_right"]) for row in pairs]
    require(all(left < right for left, right in endpoints), "Self-pair or reversed candidate pair")
    require(len(endpoints) == len(set(endpoints)), "Duplicate candidate pair")
    require(all(left in allowed_ids and right in allowed_ids for left, right in endpoints),
            "Candidate endpoint is outside its split")
    for row in pairs:
        rules = row["blocking_rules"].split("|")
        require(bool(rules) and rules == sorted(set(rules)) and set(rules) <= set(ALL_METHOD_NAMES),
                "Invalid blocking-rule provenance")


def evaluate_blocking(pairs: Sequence[Row], split_truth: Sequence[Row], split: str) -> dict:
    """Attach labels only AFTER blocking. Test performance is deliberately guarded."""
    require(split in {"train", "validation"}, "Test performance is reserved for final evaluation")
    validate_columns(split_truth, TRUTH_FIELDS, "Evaluation truth")
    validate_record_ids(split_truth, "Evaluation truth")
    mapping = {row["record_id"]: row["true_entity_id"] for row in split_truth}
    validate_candidate_pairs(pairs, set(mapping))
    counts = Counter(mapping.values())
    possible = math.comb(len(mapping), 2)
    true_pairs = sum(math.comb(count, 2) for count in counts.values())
    retrieved = sum(mapping[row["record_id_left"]] == mapping[row["record_id_right"]]
                    for row in pairs)
    return {
        "record_count": len(mapping), "entity_count": len(counts),
        "total_possible_pairs": possible, "candidate_pairs": len(pairs),
        "pair_reduction": 1 - len(pairs) / possible if possible else None,
        "all_true_duplicate_pairs": true_pairs, "true_duplicate_pairs_retrieved": retrieved,
        "missed_true_duplicate_pairs": true_pairs - retrieved,
        "candidate_match_count": retrieved, "candidate_non_match_count": len(pairs) - retrieved,
        "blocking_recall": retrieved / true_pairs if true_pairs else None,
        "candidate_pairs_by_rule": {
            rule: sum(rule in row["blocking_rules"].split("|") for row in pairs)
            for rule in ALL_METHOD_NAMES
        },
    }


def diagnose_missed_training_pairs(
    cleaned_records: Sequence[Row], truth: Sequence[Row], baseline_pairs: Sequence[Row],
    config: BlockingConfig = DEFAULT_BLOCKING_CONFIG, split: str = "train",
) -> dict:
    """Diagnostic labels identify misses only; nothing here feeds candidate retrieval."""
    require(split == "train", "Missed-pair diagnosis is restricted to training data")
    validate_columns(cleaned_records, CLEANED_FIELDS, "Training diagnosis records")
    validate_record_ids(cleaned_records, "Training diagnosis records")
    validate_columns(truth, TRUTH_FIELDS, "Training diagnosis truth")
    validate_record_ids(truth, "Training diagnosis truth")
    records = {row["record_id"]: row for row in cleaned_records}
    require(set(records) == {row["record_id"] for row in truth}, "Training diagnosis truth coverage mismatch")
    validate_candidate_pairs(baseline_pairs, set(records))
    by_entity: dict[str, list[str]] = defaultdict(list)
    for row in truth:
        by_entity[row["true_entity_id"]].append(row["record_id"])
    retrieved = {(row["record_id_left"], row["record_id_right"]) for row in baseline_pairs}
    missed = sorted(pair for members in by_entity.values()
                    for pair in combinations(sorted(members), 2) if pair not in retrieved)

    def state(left: str, right: str) -> str:
        if not left and not right:
            return "both_missing"
        if not left or not right:
            return "one_missing"
        return "equal_nonempty" if left == right else "different_nonempty"

    field_counts: dict[str, Counter] = {field: Counter() for field in DETAIL_FIELDS}
    failure_counts = {rule: Counter() for rule in RULE_NAMES}
    details = []
    for left_id, right_id in missed:
        left, right = records[left_id], records[right_id]
        states = {field: state(left[f"{field}_normalized"], right[f"{field}_normalized"])
                  for field in DETAIL_FIELDS}
        for field, status in states.items():
            field_counts[field][status] += 1
        prefixes = ["".join(char for char in row["company_name_normalized"] if char.isalnum())
                    [:config.name_prefix_length] for row in (left, right)]
        states["company_name_prefix"] = state(*prefixes)
        left_tokens, right_tokens = [set(informative_name_tokens(row["company_name_normalized"], config))
                                    for row in (left, right)]
        states["informative_name_token"] = (
            "equal_nonempty" if left_tokens & right_tokens else
            "missing_or_uninformative" if not left_tokens or not right_tokens else "no_shared_token"
        )
        failures = {
            rule: [f"{field}:{states[field]}" for field in fields if states[field] != "equal_nonempty"]
            for rule, fields in zip(RULE_NAMES, (
                ("country", "registration_id"),
                ("country", "postcode", "company_name_prefix"),
                ("country", "city", "informative_name_token"),
            ))
        }
        require(all(failures.values()), "Diagnosis inconsistent with baseline exact rules")
        for rule, reasons in failures.items():
            failure_counts[rule].update(reasons)
        details.append({"record_id_left": left_id, "record_id_right": right_id,
                        "normalized_field_states": states, "failed_rule_reasons": failures})
    return {
        "data_classification": "SYNTHETIC", "scope": "training_only",
        "missed_true_duplicate_pairs": len(missed),
        "interpretation": "Counts overlap across fields/rules. Missing values are failures, not agreements. "
                          "Diagnostic labels and pair lists are never used by candidate generation.",
        "observed_field_state_counts": {field: dict(sorted(counts.items()))
                                         for field, counts in field_counts.items()},
        "exact_rule_failure_counts": {rule: dict(sorted(counts.items()))
                                      for rule, counts in failure_counts.items()},
        "missed_pair_details": details,
    }


def load_frozen_baseline(project_root: Path, config: BlockingConfig) -> dict:
    """Validate an immutable archive; never replace it with revised outputs."""
    archive = project_root / "reports/blocking_baseline"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    for section in ("protected_input_sha256", "baseline_artifact_sha256"):
        for relative, expected_hash in manifest[section].items():
            path = project_root / relative
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash,
                    f"Frozen baseline or protected input changed: {relative}")
    baseline = json.loads((archive / "blocking_report.json").read_text(encoding="utf-8"))
    expected_config = {**asdict(config), "excluded_name_tokens": list(config.excluded_name_tokens)}
    require(baseline["blocking_configuration"] == expected_config, "Exact blocking configuration differs from baseline")
    require("fallback_configuration" not in baseline, "Baseline must precede fuzzy fallback retrieval")
    require(set(baseline["blocking_evaluation"]) == {"train", "validation"}, "Unexpected baseline evaluation scope")
    return baseline


def build_before_after_report(baseline: dict, revised: dict) -> dict:
    """Derive match/non-match counts from saved metrics, without another label join."""
    def metrics(values: dict) -> dict:
        return {
            key: values[key] for key in (
                "record_count", "entity_count", "total_possible_pairs", "candidate_pairs",
                "all_true_duplicate_pairs", "true_duplicate_pairs_retrieved", "blocking_recall", "pair_reduction"
            )
        } | {
            "missed_true_duplicate_pairs": values["all_true_duplicate_pairs"] - values["true_duplicate_pairs_retrieved"],
            "candidate_match_count": values["true_duplicate_pairs_retrieved"],
            "candidate_non_match_count": values["candidate_pairs"] - values["true_duplicate_pairs_retrieved"],
        }
    comparison = {}
    for split in ("train", "validation"):
        before = metrics(baseline["blocking_evaluation"][split])
        after = metrics(revised["blocking_evaluation"][split])
        require(all(before[key] == after[key] for key in (
            "record_count", "entity_count", "total_possible_pairs", "all_true_duplicate_pairs"
        )), "Before/after evaluation populations differ")
        comparison[split] = {"baseline": before, "revised": after, "change": {
            "additional_candidates": after["candidate_pairs"] - before["candidate_pairs"],
            "additional_true_pairs_retrieved": after["candidate_match_count"] - before["candidate_match_count"],
            "additional_non_match_candidates": after["candidate_non_match_count"] - before["candidate_non_match_count"],
        }}
    return {
        "data_classification": "SYNTHETIC", "baseline_report": "reports/blocking_baseline/blocking_report.json",
        "baseline_blocking_configuration": baseline["blocking_configuration"],
        "revised_blocking_configuration": revised["blocking_configuration"],
        "fallback_configuration": revised["fallback_configuration"],
        "evaluation_policy": "One supplied configuration evaluated on train and validation; no settings search. "
                             "Score cutoffs are retrieval settings, not final match thresholds.",
        "comparison": comparison,
        "training_diagnosis": "reports/training_missed_pairs_diagnosis.json",
        "test_performance": "Reserved for final evaluation; no test performance metrics calculated.",
    }


def run_pipeline(
    project_root: Path = PROJECT_ROOT, config: BlockingConfig = DEFAULT_BLOCKING_CONFIG,
    fallback_config: FallbackConfig = DEFAULT_FALLBACK_CONFIG,
) -> dict:
    """Run Stage 2 only; never invoke or rewrite the Stage 1 generator/inputs."""
    records = load_observed_records(project_root / "data/raw/client_records.csv")
    truth = read_csv(project_root / "data/ground_truth/record_entity_map.csv", TRUTH_FIELDS)
    parents = read_csv(project_root / "data/reference/parent_relationships.csv", PARENT_FIELDS)
    groups = build_split_groups(truth, parents)
    assignments, reused = load_or_create_splits(
        project_root / "data/splits/record_splits.csv", records, truth, groups
    )
    cleaned = clean_records(records)
    split_ids = {split: {row["record_id"] for row in assignments if row["split"] == split}
                 for split in SPLIT_NAMES}

    # Finish blocking for ALL splits before joining any labels for evaluation.
    candidates = {
        split: generate_revised_candidate_pairs(
            [row for row in cleaned if row["record_id"] in split_ids[split]], config, fallback_config)
        for split in SPLIT_NAMES
    }
    evaluation = {
        split: evaluate_blocking(candidates[split],
                                 [row for row in truth if row["record_id"] in split_ids[split]], split)
        for split in ("train", "validation")
    }
    report = {
        "data_classification": "SYNTHETIC",
        "stage": 2, "seed": SEED, "target_split_ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        "split_strategy": SPLIT_STRATEGY,
        "split_allocation_unit": "corporate-family connected component (including independent entities)",
        "blocking_configuration": {**asdict(config),
                                   "excluded_name_tokens": list(config.excluded_name_tokens)},
        "fallback_configuration": {**asdict(fallback_config), "scorer": "WRatio",
                                   "rapidfuzz_version": version("rapidfuzz"),
                                   "tie_break": "score descending, record_id ascending",
                                   "scope": "within each split; no country/city restriction"},
        "zero_denominator_policy": "null (undefined) for pair reduction or recall with zero denominator",
        "split_counts": {
            split: {"records": len(split_ids[split]),
                    "entities": len({row["true_entity_id"] for row in assignments if row["split"] == split}),
                    "corporate_family_groups": len({row["split_group_id"] for row in assignments if row["split"] == split})}
            for split in SPLIT_NAMES
        },
        "blocking_evaluation": evaluation,
        "test_performance": "Reserved for final evaluation; no test performance metrics calculated.",
        "validation": {
            "record_entity_family_split_isolation": "passed",
            "candidate_pair_uniqueness_and_within_split_endpoints": "passed",
            "observed_only_cleaning_and_blocking_inputs": "passed",
        },
    }
    write_csv(project_root / "data/processed/cleaned_records.csv", CLEANED_FIELDS, cleaned)
    for split in SPLIT_NAMES:
        write_csv(project_root / f"data/processed/candidate_pairs_{split}.csv", PAIR_FIELDS, candidates[split])
    report_path = project_root / "reports/blocking_report.json"
    write_json(report_path, report)

    print("SYNTHETIC DATA ONLY - Stage 2 cleaning, family splits, and candidate blocking")
    print(f"Split assignment: {'reused' if reused else 'created'} (seed {SEED})")
    for split, counts in report["split_counts"].items():
        print(f"{split}: {counts['records']} records, {counts['entities']} entities, "
              f"{counts['corporate_family_groups']} corporate-family groups")
    for split, metrics in evaluation.items():
        recall = metrics["blocking_recall"]
        reduction = metrics["pair_reduction"]
        print(f"{split}: {metrics['candidate_pairs']} / {metrics['total_possible_pairs']} possible pairs; "
              f"retrieved {metrics['true_duplicate_pairs_retrieved']} / "
              f"{metrics['all_true_duplicate_pairs']} true duplicate pairs")
        print(f"  pair reduction: {format(reduction, '.2%') if reduction is not None else 'undefined'}; "
              f"blocking recall: {format(recall, '.2%') if recall is not None else 'undefined'}")
    print("Test candidates saved; test performance reserved for final evaluation.")
    print("PASS: record/entity/family isolation; unique within-split pairs; observed-only blocking")
    print("Saved cleaned records, split assignment, three candidate files, and reports/blocking_report.json")
    return report


def run_improvement(
    project_root: Path = PROJECT_ROOT, config: BlockingConfig = DEFAULT_BLOCKING_CONFIG,
    fallback_config: FallbackConfig = DEFAULT_FALLBACK_CONFIG,
) -> dict:
    baseline = load_frozen_baseline(project_root, config)
    require((project_root / "data/splits/record_splits.csv").is_file(), "Existing splits are required")
    records = load_observed_records(project_root / "data/raw/client_records.csv")
    truth = read_csv(project_root / "data/ground_truth/record_entity_map.csv", TRUTH_FIELDS)
    assignments = read_csv(project_root / "data/splits/record_splits.csv", SPLIT_FIELDS)
    train_ids = {row["record_id"] for row in assignments if row["split"] == "train"}
    training = clean_records([row for row in records if row["record_id"] in train_ids])
    baseline_train = read_csv(project_root / "data/processed/baseline/candidate_pairs_train.csv", PAIR_FIELDS)
    diagnosis = diagnose_missed_training_pairs(
        training, [row for row in truth if row["record_id"] in train_ids], baseline_train, config
    )
    expected = baseline["blocking_evaluation"]["train"]
    require(diagnosis["missed_true_duplicate_pairs"] ==
            expected["all_true_duplicate_pairs"] - expected["true_duplicate_pairs_retrieved"],
            "Training diagnosis differs from frozen baseline")
    write_json(project_root / "reports/training_missed_pairs_diagnosis.json", diagnosis)

    # The only revised evaluation pass; no settings sweep or test-label evaluation.
    revised = run_pipeline(project_root, config, fallback_config)
    for split in SPLIT_NAMES:
        old_pairs = read_csv(project_root / f"data/processed/baseline/candidate_pairs_{split}.csv", PAIR_FIELDS)
        new_pairs = read_csv(project_root / f"data/processed/candidate_pairs_{split}.csv", PAIR_FIELDS)
        methods = {(row["record_id_left"], row["record_id_right"]): set(row["blocking_rules"].split("|"))
                   for row in new_pairs}
        require(all(set(row["blocking_rules"].split("|")) <=
                    methods.get((row["record_id_left"], row["record_id_right"]), set()) for row in old_pairs),
                f"Revised {split} pairs must preserve every baseline pair and its provenance")
    comparison = build_before_after_report(baseline, revised)
    write_json(project_root / "reports/blocking_comparison.json", comparison)
    print(f"Training-only baseline diagnosis: {diagnosis['missed_true_duplicate_pairs']} missed true pairs")
    print("PASS: baseline pairs/provenance retained in every split; frozen baseline verified")
    print("Saved reports/training_missed_pairs_diagnosis.json and reports/blocking_comparison.json")
    return comparison


if __name__ == "__main__":
    run_improvement()
