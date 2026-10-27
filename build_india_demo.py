"""Build one separate Indian teaching dataset and score it with the frozen model.

Real public company names; simulated records, errors, identifiers and addresses.
No training, new evaluation metrics, retrieval tuning or truth-based graph repair.
The entry point refuses to overwrite an existing run. --verify only checks hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from dataclasses import asdict
from pathlib import Path

import build_clusters as bc
import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm
from portfolio_ui import artifacts as ad


ROOT = Path(__file__).resolve().parent
DATA_DIR = "data/india_demo"
OUTPUT_DIR = "outputs/india_demo/v1"
RETRIEVED_ON = "2026-10-26"
DISCLAIMER = ("Real public company names; simulated source records, errors, addresses, PIN codes and DEMO-IN identifiers. "
              "No actual customer data, genuine bank records, affiliation or endorsement.")
COMPANY_FIELDS = ("reference_entity_id", "company_name", "source_url", "retrieved_on", "verification_note")
RELATIONSHIP_FIELDS = ("parent_entity_id", "child_entity_id", "relationship_type", "relationship_source", "source_url", "retrieved_on")
PREDICTION_FIELDS = (*pc.PAIR_FIELDS, *pf.FEATURE_NAMES, "model_score", "predicted_match")
SCENARIO_LABELS = (
    "Same entity with spelling differences", "Same entity with missing details",
    "Different entities", "Related but separate companies",
)

# Only names and the one supplied relationship are public facts. None of the
# addresses/IDs below are copied from the public sites or asserted to be genuine.
SPECS = (
    ("PUBLIC-IN-SBI", "State Bank of India", "State Bnak of India", "SBI", "Mumbai", "499901",
     "101 Demo Learning Road, Sample Finance Park", "DEMO-IN-7041",
     "https://sbi.bank.in/en/web/customer-care/contact-us", "Official contact page names State Bank of India."),
    ("PUBLIC-IN-HDFC", "HDFC Bank Limited", "HDFC Bnak Ltd.", "HDFC BANK", "Mumbai", "499902",
     "202 Demo Market Road, Sample Commerce Block", "DEMO-IN-8152",
     "https://www.hdfc.bank.in/about-us/who-we-are", "Official overview states the incorporated name HDFC Bank Limited."),
    ("PUBLIC-IN-ICICI", "ICICI Bank Limited", "ICICI Bnak Limited", "ICICI BANK LTD", "Hyderabad", "599903",
     "303 Demo Knowledge Avenue, Sample Business Nagar", "DEMO-IN-9263",
     "https://www.icicibank.hk/about_us", "Official bank page identifies ICICI Bank Limited as incorporated in India."),
    ("PUBLIC-IN-AXIS", "Axis Bank Limited", "Axis Bnak Ltd.", "AXIS BANK", "Ahmedabad", "399904",
     "404 Demo River Road, Sample Accounts Park", "DEMO-IN-1374",
     "https://www.axis.bank.in/shareholders-corner/investor-faqs", "Official investor FAQs use the name Axis Bank Limited."),
    ("PUBLIC-IN-BAJAJ-FINANCE", "Bajaj Finance Limited", "Bajaj Fianance Ltd.", "BAJAJ FINANCE", "Pune", "499905",
     "505 Demo Finance Road, Sample Learning Park", "DEMO-IN-2485",
     "https://www.aboutbajajfinserv.com/finance-about-us", "Official company page names Bajaj Finance Limited and describes its parent."),
    ("PUBLIC-IN-BAJAJ-FINSERV", "Bajaj Finserv Limited", "Bajaj Fisnerv Ltd.", "BAJAJ FINSERV", "Pune", "499905",
     "505 Demo Finance Road, Sample Learning Park", "DEMO-IN-3596",
     "https://www.aboutbajajfinserv.com/about-us", "Official company overview identifies Bajaj Finserv Limited."),
)


def make_demo_records():
    """A fixed teaching fixture, decided before scoring; IDs are opaque and seeded."""
    rng = random.Random(42)
    records, truth, companies, grouped_ids = [], [], [], []
    for index, (reference_id, name, typo, abbreviation, city, pin, address, fake_id, url, note) in enumerate(SPECS):
        companies.append(dict(zip(COMPANY_FIELDS, (reference_id, name, url, RETRIEVED_ON, note))))
        variants = [
            ("DEMO_CORE", name, fake_id, address, city, pin, "India"),
            ("DEMO_CRM", typo, fake_id if index != 5 else "DEMO-IN-3569",
             address.replace("Road", "Rd.").replace("Avenue", "Ave."), city.upper(), pin, "INDIA"),
            ("DEMO_OPERATIONS", abbreviation, fake_id, address.replace("Road", "Rd").replace("Avenue", "Ave"), city, pin,
             "" if index == 0 else "India"),
            ("DEMO_LEGACY", name.replace(" Limited", "") if index != 3 else "", "",
             "" if index != 3 else "404 Demo Rvr Rd, Sample Accounts Park", city, "", "India"),
        ]
        ids = []
        for source, company_name, registration, observed_address, observed_city, postcode, country in variants:
            record_id = "in_rec_" + uuid.UUID(int=rng.getrandbits(128), version=4).hex
            records.append(dict(zip(pc.OBSERVED_FIELDS, (record_id, source, company_name, registration,
                                                       observed_address, observed_city, postcode, country))))
            truth.append({"record_id": record_id, "true_entity_id": f"DEMO-ENTITY-{index + 1:02d}"})
            ids.append(record_id)
        grouped_ids.append(ids)
    relationships = [{"parent_entity_id": "PUBLIC-IN-BAJAJ-FINSERV", "child_entity_id": "PUBLIC-IN-BAJAJ-FINANCE",
                      "relationship_type": "parent_to_subsidiary", "relationship_source": "Bajaj official company overview",
                      "source_url": "https://www.aboutbajajfinserv.com/finance-about-us", "retrieved_on": RETRIEVED_ON}]
    # These pointers label educational examples only; they never enter retrieval,
    # features, scoring or graph construction. The labels are checked after scoring.
    scenario_pairs = ((grouped_ids[1][0], grouped_ids[1][1]), (grouped_ids[2][0], grouped_ids[2][3]),
                      (grouped_ids[0][0], grouped_ids[3][0]), (grouped_ids[5][0], grouped_ids[4][0]))
    scenarios = [{"scenario_id": f"example-{index + 1}", "label": label,
                  "record_id_left": pair[0], "record_id_right": pair[1]}
                 for index, (label, pair) in enumerate(zip(SCENARIO_LABELS, scenario_pairs))]
    return records, truth, companies, relationships, scenarios


def predict_observed_records(records, pipeline, threshold):
    """This boundary accepts observed fields only. No truth/reference arguments."""
    pc.validate_columns(records, pc.OBSERVED_FIELDS, "Indian observed demo")
    pc.validate_record_ids(records, "Indian observed demo")
    cleaned = pc.clean_records(records)
    pairs = pc.generate_revised_candidate_pairs(cleaned)
    projected = {row["record_id"]: {field: row[field] for field in pf.NORMALIZED_FIELDS} for row in cleaned}
    features = pf.build_feature_frame(pairs, projected)
    scores, predictions = tm.predict_logistic(pipeline, features, threshold)
    by_id = {row["record_id"]: row for row in cleaned}
    graph = bc.build_duplicate_graph(by_id, pairs, scores, threshold)
    membership, clusters = bc.cluster_memberships(graph)
    flags = bc.registration_review_flags(clusters, by_id)
    summaries = bc.cluster_summaries(graph, clusters, flags)
    return {"cleaned": cleaned, "pairs": pairs, "features": features, "scores": scores, "predictions": predictions,
            "records": by_id, "graph": graph, "membership": membership, "clusters": clusters, "flags": flags, "summaries": summaries}


def files_under(root, directory):
    return {path.relative_to(root).as_posix(): ad.digest(path) for path in sorted((root / directory).rglob("*")) if path.is_file()}


def verify_run(root=ROOT):
    path = root / OUTPUT_DIR / "prediction_run.json"
    pc.require(path.is_file(), "Indian demo outputs missing; run build_india_demo.py once")
    run = ad.read_json(path)
    for relative, expected in {**run["data_sha256"], **run["prediction_files_sha256"], **run["frozen_input_sha256"]}.items():
        pc.require(ad.digest(root / relative) == expected, f"Indian demo/frozen artifact changed: {relative}")
    return run


def build_demo(root=ROOT):
    data_path, output_path = root / DATA_DIR, root / OUTPUT_DIR
    pc.require(not data_path.exists() and not output_path.exists(), "Indian demo already exists. Use --verify; no overwrite or tuning rerun.")
    catalog = ad.load_catalog(root)
    frozen = catalog.freeze
    pc.require(json.loads(json.dumps(asdict(pc.DEFAULT_BLOCKING_CONFIG))) == frozen["blocking"]["exact"],
               "Existing exact blocking defaults disagree with the freeze")
    pc.require(all(frozen["blocking"]["fallback"][key] == value for key, value in asdict(pc.DEFAULT_FALLBACK_CONFIG).items()),
               "Existing fallback defaults disagree with the freeze")
    frozen_hashes = {relative: ad.digest(root / relative) for relative in frozen["protected_sha256"]}
    pipeline, threshold = ad.load_frozen_model(catalog)
    records, truth_rows, companies, relationships, scenarios = make_demo_records()
    pc.write_csv(data_path / "observed/client_records.csv", pc.OBSERVED_FIELDS, records)
    pc.write_csv(data_path / "reference/companies.csv", COMPANY_FIELDS, companies)
    pc.write_csv(data_path / "reference/public_relationships.csv", RELATIONSHIP_FIELDS, relationships)

    # A fresh candidate search and graph on the Indian observations, never copied
    # or relabelled from the v1 dataset. There is exactly one batch prediction run.
    result = predict_observed_records(records, pipeline, threshold)
    pc.write_csv(output_path / "cleaned_records.csv", pc.CLEANED_FIELDS, result["cleaned"])
    pc.write_csv(output_path / "candidate_pairs.csv", pc.PAIR_FIELDS, result["pairs"])
    predicted_rows = [{**pair, **result["features"].iloc[index].to_dict(), "model_score": float(result["scores"][index]),
                       "predicted_match": int(result["predictions"][index])} for index, pair in enumerate(result["pairs"])]
    pc.write_csv(output_path / "candidate_predictions.csv", PREDICTION_FIELDS, predicted_rows)
    bc.save_prediction_outputs(output_path / "clusters", result["graph"], result["records"], result["membership"],
                               result["clusters"], result["flags"], result["summaries"])

    # Truth is written and consulted for teaching labels only AFTER predictions.
    pc.write_csv(data_path / "ground_truth/record_entity_map.csv", pc.TRUTH_FIELDS, truth_rows)
    truth = {row["record_id"]: row["true_entity_id"] for row in truth_rows}
    for scenario in scenarios:
        same = truth[scenario["record_id_left"]] == truth[scenario["record_id_right"]]
        scenario["expected_outcome"] = "same legal entity" if same else "different legal entities"
        scenario["expectation_source"] = "simulation ground truth; educational label, not a model prediction"
    pc.write_json(data_path / "education/guided_examples.json", {"examples": scenarios})
    pc.write_json(data_path / "dataset_info.json", {
        "dataset_id": "india_demo_v1", "seed": 42, "disclaimer": DISCLAIMER,
        "record_count": len(records), "simulated_entity_count": len(companies), "official_sources_retrieved_on": RETRIEVED_ON,
        "identifier_policy": "All nonempty registration identifiers start DEMO-IN; none are genuine CINs, IFSCs or account IDs",
        "address_policy": "Invented demonstration addresses and six-digit unverified synthetic PIN codes in Indian cities",
        "truth_policy": "Separate educational labels, never used for candidates, model features or cluster repair",
    })
    run = {
        "dataset_id": "india_demo_v1", "disclaimer": DISCLAIMER, "batch_prediction_runs": 1,
        "model_path": catalog.choice["model_path"], "model_sha256": frozen["selected_model_sha256"],
        "original_freeze_sha256": catalog.freeze_sha256, "feature_schema": list(pf.FEATURE_NAMES), "threshold": threshold,
        "blocking_configuration": frozen["blocking"],
        "clustering": {**frozen["clustering"], "node_scope": "every observed record in the separate Indian demo; no original v1 records"},
        "review_flags": frozen["review_flags"],
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": {"records": len(records), "candidates": len(result["pairs"]), "direct_edges": result["graph"].number_of_edges(),
                   "predicted_groups": len(result["clusters"]), "singletons": sum(len(ids) == 1 for ids in result["clusters"].values()),
                   "review_flags": len(result["flags"])},
        "performance": "NOT EVALUATED: descriptive prediction counts only; no Indian accuracy/precision/recall claims",
        "truth_used_for_predictions": False, "reference_relationships_used_for_predictions": False,
        "data_sha256": files_under(root, DATA_DIR), "prediction_files_sha256": files_under(root, OUTPUT_DIR),
        "frozen_input_sha256": frozen_hashes,
    }
    pc.write_json(output_path / "prediction_run.json", run)
    verify_run(root)
    print(DISCLAIMER)
    print(json.dumps(run["counts"], indent=2))
    print("One unchanged-model batch run complete. Indian accuracy not evaluated. Original v1 artifacts unchanged.")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check saved hashes only; do not rerun predictions")
    arguments = parser.parse_args()
    if arguments.verify:
        verified = verify_run()
        print("PASS: Indian demo and frozen input hashes unchanged", verified["counts"])
    else:
        build_demo()
