"""Stage 1: generate reproducible SYNTHETIC banking client data.

Every company, source record, and relationship in this project is fictional.
No matching model, candidate pairs, or application is built in this stage.
"""

from __future__ import annotations

import csv
import random
import re
import string
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from faker import Faker


SEED = 42
ENTITY_COUNT = 300
SOURCE_SYSTEMS = (
    "ASTER_CRM",
    "BIRCH_KYC",
    "CEDAR_LENDING",
    "DUNE_PAYMENTS",
    "ELM_TREASURY",
)

# Resolve paths from this file, not from the terminal's current directory.
# pathlib handles Windows paths and folder names that contain spaces.
PROJECT_ROOT = Path(__file__).resolve().parent
RECORDS_PATH = PROJECT_ROOT / "data" / "raw" / "client_records.csv"
MAP_PATH = PROJECT_ROOT / "data" / "ground_truth" / "record_entity_map.csv"
ENTITIES_PATH = PROJECT_ROOT / "data" / "ground_truth" / "entities.csv"
PARENTS_PATH = PROJECT_ROOT / "data" / "reference" / "parent_relationships.csv"

DETAIL_FIELDS = (
    "company_name", "registration_id", "address", "city", "postcode", "country"
)
RECORD_FIELDS = ("record_id", "source_system", *DETAIL_FIELDS)
ENTITY_FIELDS = ("true_entity_id", *DETAIL_FIELDS)
MAP_FIELDS = ("record_id", "true_entity_id")
PARENT_FIELDS = ("child_entity_id", "parent_entity_id", "relationship_source")

FIRST_SYLLABLES = (
    "al", "bel", "cor", "dan", "el", "fen", "gal", "har", "kel", "lin",
    "mer", "nor", "or", "pen", "rav", "sel", "tor", "val", "wes", "zel",
)
MIDDLE_SYLLABLES = ("a", "e", "i", "o", "u", "en", "ar", "or", "vi", "lo")
LAST_SYLLABLES = (
    "brook", "crest", "dale", "ford", "mere", "mont", "nex", "port", "vale", "well"
)
BUSINESS_TERMS = (
    "Advisory", "Capital", "Consulting", "Industries", "Logistics",
    "Partners", "Resources", "Solutions", "Technology", "Trading",
)
ADDRESS_ABBREVIATIONS = {
    "street": "St", "road": "Rd", "avenue": "Ave", "lane": "Ln",
    "drive": "Dr", "court": "Ct", "place": "Pl", "square": "Sq",
    "terrace": "Ter", "flat": "Apt", "floor": "Fl",
}
MISSING_PROBABILITIES = {
    "company_name": 0.03,
    "registration_id": 0.12,
    "address": 0.10,
    "city": 0.06,
    "postcode": 0.08,
    "country": 0.02,
}

Row = dict[str, str]


def single_typo(value: str, rng: random.Random) -> str:
    """Introduce one deletion, substitution, or adjacent-letter transposition."""
    positions = [i for i, char in enumerate(value) if char in string.ascii_letters]
    if not positions:
        return value
    position = rng.choice(positions)
    operation = rng.choice(("delete", "replace", "transpose"))
    if operation == "delete":
        return value[:position] + value[position + 1:]
    if (
        operation == "transpose"
        and position + 1 < len(value)
        and value[position + 1] in string.ascii_letters
        and value[position] != value[position + 1]
    ):
        return (
            value[:position] + value[position + 1] + value[position]
            + value[position + 2:]
        )
    # A replacement also provides a fallback when transposition is impossible.
    alternatives = string.ascii_lowercase.replace(value[position].lower(), "")
    replacement = rng.choice(alternatives)
    if value[position].isupper():
        replacement = replacement.upper()
    return value[:position] + replacement + value[position + 1:]


def vary_case(value: str, rng: random.Random) -> str:
    if rng.random() < 0.40:
        return rng.choice((str.upper, str.lower, str.title))(value)
    return value


def create_entities(fake: Faker, rng: random.Random) -> tuple[list[Row], list[str]]:
    """Create 300 separate legal entities with complete canonical details."""
    entities: list[Row] = []
    brands: list[str] = []
    used_brands: set[str] = set()
    used_registrations: set[str] = set()

    for number in range(1, ENTITY_COUNT + 1):
        # Invented brands reduce accidental resemblance to real companies.
        while True:
            brand = (
                rng.choice(FIRST_SYLLABLES)
                + rng.choice(MIDDLE_SYLLABLES)
                + rng.choice(LAST_SYLLABLES)
            ).capitalize()
            if brand not in used_brands:
                used_brands.add(brand)
                break
        while True:
            registration = str(rng.randrange(10_000_000, 100_000_000))
            if registration not in used_registrations:
                used_registrations.add(registration)
                break
        brands.append(brand)
        entities.append({
            "true_entity_id": f"ENT_{number:04d}",
            "company_name": f"{brand} {rng.choice(BUSINESS_TERMS)} Limited",
            "registration_id": registration,
            "address": ", ".join(fake.street_address().splitlines()),
            "city": fake.city(),
            "postcode": fake.postcode(),
            "country": "United Kingdom",
        })
    return entities, brands


def add_difficult_non_matches(
    entities: list[Row], brands: list[str], rng: random.Random
) -> None:
    """Similar names and shared offices do not imply the same legal entity."""
    used_brands = set(brands)
    # Entities 1-24 form 12 similar-name pairs with different registrations.
    for first in range(0, 24, 2):
        while True:
            similar_brand = single_typo(brands[first], rng)
            if similar_brand.casefold() not in {b.casefold() for b in used_brands}:
                used_brands.add(similar_brand)
                break
        descriptor = entities[first]["company_name"][len(brands[first]):]
        entities[first + 1]["company_name"] = similar_brand + descriptor

    # Entities 25-48 form eight unrelated groups sharing a full postal address.
    for start in range(24, 48, 3):
        for member in (start + 1, start + 2):
            for field in ("address", "city", "postcode", "country"):
                entities[member][field] = entities[start][field]


def create_parent_relationships(entities: list[Row], brands: list[str]) -> list[Row]:
    """Supply 12 fictional corporate families, including a grandchild in each."""
    relationships: list[Row] = []
    roles = ("Holdings", "Services", "Technology", "Trading", "Regional Services")
    # These 60 entities are separate from the deliberate unrelated lookalikes.
    for start in range(48, 108, 5):
        for offset, role in enumerate(roles):
            entities[start + offset]["company_name"] = f"{brands[start]} {role} Limited"
        # A parent and subsidiary can also share an office while remaining distinct.
        for field in ("address", "city", "postcode", "country"):
            entities[start + 1][field] = entities[start][field]
        # Edges run parent -> child in the validation graph. The last is a grandchild.
        for child_offset, parent_offset in ((1, 0), (2, 0), (3, 0), (4, 1)):
            relationships.append({
                "child_entity_id": entities[start + child_offset]["true_entity_id"],
                "parent_entity_id": entities[start + parent_offset]["true_entity_id"],
                "relationship_source": "SYNTHETIC_CORPORATE_REGISTRY",
            })
    return relationships


def make_observed_details(entity: Row, rng: random.Random) -> Row:
    """Copy canonical details, then simulate ordinary source-system data quality."""
    observed = {field: entity[field] for field in DETAIL_FIELDS}
    name = observed["company_name"]
    if rng.random() < 0.60:
        name = re.sub(r"\bLimited$", rng.choice(("Ltd", "Ltd.")), name)
    if rng.random() < 0.25:
        name = single_typo(name, rng)
    if rng.random() < 0.25:
        name = re.sub(r"\s+(Limited|Ltd\.?)$", r", \1", name)
    observed["company_name"] = vary_case(name, rng)

    address = observed["address"]
    if rng.random() < 0.65:
        for word, abbreviation in ADDRESS_ABBREVIATIONS.items():
            address = re.sub(rf"\b{word}\b", abbreviation, address, flags=re.IGNORECASE)
    if rng.random() < 0.25:
        address = address.replace(",", "")
    if rng.random() < 0.08:
        address = single_typo(address, rng)
    observed["address"] = vary_case(address, rng)
    observed["city"] = vary_case(observed["city"], rng)
    if rng.random() < 0.04:
        observed["city"] = single_typo(observed["city"], rng)

    if rng.random() < 0.30:
        observed["postcode"] = observed["postcode"].replace(" ", "")
    if rng.random() < 0.20:
        observed["postcode"] = observed["postcode"].lower()
    if rng.random() < 0.25:
        observed["country"] = rng.choice(("UK", "U.K."))

    # Even registration numbers can contain a transcription error or formatting.
    registration = observed["registration_id"]
    if rng.random() < 0.03:
        position = rng.randrange(len(registration))
        digit = rng.choice(string.digits.replace(registration[position], ""))
        registration = registration[:position] + digit + registration[position + 1:]
    if rng.random() < 0.12:
        registration = registration[:4] + " " + registration[4:]
    observed["registration_id"] = registration

    # Empty CSV cells represent missing data; IDs and source names stay populated.
    for field, probability in MISSING_PROBABILITIES.items():
        if rng.random() < probability:
            observed[field] = ""
    return observed


def create_source_records(
    entities: list[Row], rng: random.Random
) -> tuple[list[Row], list[Row]]:
    records: list[Row] = []
    truth_map: list[Row] = []
    record_counts = [rng.randint(1, 4) for _ in entities]
    for entity, count in zip(entities, record_counts):
        # Sampling without replacement gives at most one record per source/entity.
        for source in rng.sample(SOURCE_SYSTEMS, count):
            # Seeded random UUID bits are opaque: no entity, source, or row number.
            # They are reproducible dataset identifiers, not security tokens.
            record_id = "rec_" + uuid.UUID(int=rng.getrandbits(128), version=4).hex
            records.append({
                "record_id": record_id,
                "source_system": source,
                **make_observed_details(entity, rng),
            })
            truth_map.append({
                "record_id": record_id,
                "true_entity_id": entity["true_entity_id"],
            })
    # Remove entity grouping from file order as well as from record identifiers.
    rng.shuffle(records)
    rng.shuffle(truth_map)
    return records, truth_map


def require(condition: bool, message: str) -> None:
    """Fail clearly even if Python runs with assertions disabled (-O)."""
    if not condition:
        raise ValueError(f"Dataset validation failed: {message}")


def validate_data(
    records: list[Row], truth_map: list[Row], entities: list[Row], parents: list[Row]
) -> nx.DiGraph:
    """Validate observed data, truth coverage, and the supplied corporate hierarchy."""
    for rows, fields in (
        (records, RECORD_FIELDS), (truth_map, MAP_FIELDS),
        (entities, ENTITY_FIELDS), (parents, PARENT_FIELDS),
    ):
        require(all(set(row) == set(fields) for row in rows), "unexpected CSV columns")

    entity_ids = {row["true_entity_id"] for row in entities}
    require(len(entities) == len(entity_ids) == ENTITY_COUNT, "expected 300 unique entities")
    require(all(all(row[field] for field in ENTITY_FIELDS) for row in entities),
            "missing canonical entity details")
    require(len({row["registration_id"] for row in entities}) == ENTITY_COUNT,
            "canonical registration IDs must be distinct")
    require(len({row["company_name"].casefold() for row in entities}) == ENTITY_COUNT,
            "canonical company names must be distinct")

    record_ids = {row["record_id"] for row in records}
    mapped_ids = {row["record_id"] for row in truth_map}
    require(len(records) == len(record_ids) and all(record_ids), "record IDs not unique/populated")
    require(len(truth_map) == len(mapped_ids), "duplicate truth mappings")
    require(record_ids == mapped_ids, "missing or orphaned truth mappings")
    require({row["true_entity_id"] for row in truth_map} == entity_ids,
            "truth must reference all and only canonical entities")
    counts = Counter(row["true_entity_id"] for row in truth_map)
    require(all(1 <= count <= 4 for count in counts.values()), "record count outside 1-4")
    require(any(count == 1 for count in counts.values()), "expected singleton entities")
    mapping = {row["record_id"]: row["true_entity_id"] for row in truth_map}
    require(all(row["source_system"] in SOURCE_SYSTEMS for row in records),
            "unknown source system")
    require(len({(mapping[row["record_id"]], row["source_system"]) for row in records})
            == len(records), "duplicate source records for an entity")
    require(all(any(row[field] for field in DETAIL_FIELDS) for row in records),
            "observed record has no client details")
    require(any(row[field] == "" for row in records for field in DETAIL_FIELDS),
            "expected missing values in observed records")

    edges = [(row["parent_entity_id"], row["child_entity_id"]) for row in parents]
    require(all(parent in entity_ids and child in entity_ids for parent, child in edges),
            "invalid parent/child reference")
    require(all(parent != child for parent, child in edges), "self-parenting")
    require(len(edges) == len(set(edges)), "duplicate hierarchy relationships")
    require(all(row["relationship_source"] == "SYNTHETIC_CORPORATE_REGISTRY"
                for row in parents), "missing synthetic relationship provenance")
    graph = nx.DiGraph()
    graph.add_nodes_from(row["true_entity_id"] for row in entities)
    graph.add_edges_from(edges)
    require(nx.is_directed_acyclic_graph(graph), "hierarchy contains a cycle")
    require(all(degree <= 1 for _, degree in graph.in_degree()), "multiple direct parents")
    require(bool(edges) and any(degree == 0 for _, degree in graph.degree()),
            "expected both corporate families and independent entities")
    return graph


def write_csv(path: Path, fields: tuple[str, ...], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[Row]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_summary(
    records: list[Row], truth_map: list[Row], entities: list[Row],
    parents: list[Row], graph: nx.DiGraph,
) -> None:
    print("SYNTHETIC DATA ONLY - fictional clients, systems, and corporate relationships")
    print(f"Seed: {SEED}")
    print(f"Canonical legal entities: {len(entities)}")
    print(f"Observed source records: {len(records)}")
    print(f"Truth mappings: {len(truth_map)}")
    print(f"Source systems represented: {len({row['source_system'] for row in records})}")
    distribution = Counter(Counter(row["true_entity_id"] for row in truth_map).values())
    for count in range(1, 5):
        print(f"Entities with {count} source record(s): {distribution[count]}")
    families = list(nx.weakly_connected_components(graph))
    print(f"Supplied parent-subsidiary relationships: {len(parents)}")
    print(f"Corporate families with multiple legal entities: {sum(len(c) > 1 for c in families)}")
    print(f"Independent entities: {sum(len(c) == 1 for c in families)}")
    print(f"Maximum hierarchy depth (edges): {nx.dag_longest_path_length(graph)}")
    addresses: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for entity in entities:
        key = tuple(entity[field] for field in ("address", "city", "postcode", "country"))
        addresses[key].append(entity["true_entity_id"])
    shared = [ids for ids in addresses.values() if len(ids) > 1]
    print(f"Shared-address groups of separate legal entities: {len(shared)}")
    print(f"Shared-address groups of independent entities: "
          f"{sum(all(graph.degree(entity_id) == 0 for entity_id in ids) for ids in shared)}")
    missing = ", ".join(
        f"{field}={sum(row[field] == '' for row in records)}" for field in DETAIL_FIELDS
    )
    print(f"Missing observed values: {missing}")
    print("PASS: 300 distinct canonical entities; 1-4 records per entity; singletons present")
    print("PASS: unique opaque record IDs; complete one-to-one truth mappings")
    print("PASS: observed records contain only the eight allowed fields; truth labels excluded")
    print("PASS: valid parent references; no self-parenting or duplicate relationships")
    print("PASS: acyclic hierarchy; at most one direct parent; independent entities present")
    print("PASS: saved CSV files reloaded and validated successfully")
    for path in (RECORDS_PATH, MAP_PATH, ENTITIES_PATH, PARENTS_PATH):
        print(f"Saved: {path.relative_to(PROJECT_ROOT).as_posix()}")


def main() -> None:
    # Local generators keep runs independent from other code's random state.
    rng = random.Random(SEED)
    fake = Faker("en_GB")
    fake.seed_instance(SEED)
    entities, brands = create_entities(fake, rng)
    add_difficult_non_matches(entities, brands, rng)
    parents = create_parent_relationships(entities, brands)
    records, truth_map = create_source_records(entities, rng)

    # Validate before writing; then validate what was actually saved to disk.
    validate_data(records, truth_map, entities, parents)
    for path, fields, rows in (
        (RECORDS_PATH, RECORD_FIELDS, records),
        (MAP_PATH, MAP_FIELDS, truth_map),
        (ENTITIES_PATH, ENTITY_FIELDS, entities),
        (PARENTS_PATH, PARENT_FIELDS, parents),
    ):
        write_csv(path, fields, rows)
    saved_records = read_csv(RECORDS_PATH)
    saved_map = read_csv(MAP_PATH)
    saved_entities = read_csv(ENTITIES_PATH)
    saved_parents = read_csv(PARENTS_PATH)
    graph = validate_data(saved_records, saved_map, saved_entities, saved_parents)
    print_summary(saved_records, saved_map, saved_entities, saved_parents, graph)


if __name__ == "__main__":
    main()
