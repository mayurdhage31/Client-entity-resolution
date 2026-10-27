"""Ephemeral observed-only direct pair scoring. No caching, logging or writes."""

import pair_features as pf
import prepare_candidates as pc
import train_matcher as tm


FIELDS = pc.DETAIL_FIELDS
FIELD_LABELS = {"company_name": "Company name", "registration_id": "Registration ID", "address": "Address",
                "city": "City", "postcode": "Postcode", "country": "Country"}


def normalize_entered_record(values, side):
    if set(values) != set(FIELDS) or any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Live scoring accepts exactly six observed text fields; truth labels and record IDs are excluded.")
    if not any(value.strip() for value in values.values()):
        raise ValueError(f"Record {side} is empty. Enter at least one observed field in the optional editable copy.")
    # Synthetic local IDs only locate these two values; they never enter features.
    observed = {"record_id": f"interactive_{side}", "source_system": "INTERACTIVE_SYNTHETIC_DEMO", **values}
    normalized = pc.clean_records([observed])[0]
    projected = {field: normalized[field] for field in pf.NORMALIZED_FIELDS}
    if not any(projected.values()):
        raise ValueError(f"Record {side} has no nonempty text after cleaning. Enter an observed field with letters or digits.")
    return projected


def score_records(pipeline, threshold, left, right):
    a, b = normalize_entered_record(left, "A"), normalize_entered_record(right, "B")
    pairs = [{"record_id_left": "left", "record_id_right": "right"}]
    features = pf.build_feature_frame(pairs, {"left": a, "right": b})
    scores, predictions = tm.predict_logistic(pipeline, features, threshold)
    return {"score": float(scores[0]), "prediction": bool(predictions[0]), "threshold": threshold,
            "features": features.iloc[0].to_dict(), "left_normalized": a, "right_normalized": b,
            "left_observed": dict(left), "right_observed": dict(right)}


def evidence_rows(result):
    rows = []
    features = result["features"]
    for field in FIELDS:
        left, right = result["left_normalized"][field + "_normalized"], result["right_normalized"][field + "_normalized"]
        missing = "Both missing: no agreement" if not left and not right else "A missing" if not left else "B missing" if not right else "Both present"
        evidence = {"Field": FIELD_LABELS[field], "Normalized A": left, "Normalized B": right, "Availability": missing}
        if field in ("company_name", "address"):
            evidence["Ratio (0–1)"] = features[field + "_ratio"]
            evidence["Token-sort (0–1)"] = features[field + "_token_sort_ratio"]
        else:
            evidence["Nonempty exact agreement"] = "Yes" if features[field + "_exact"] else "No"
        rows.append(evidence)
    return rows
