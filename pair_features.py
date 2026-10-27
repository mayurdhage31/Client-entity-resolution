"""Observed-only, symmetric numeric features for SYNTHETIC company pairs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# This is the complete model-input allowlist, in its saved/prediction order.
FEATURE_NAMES = (
    "company_name_ratio", "company_name_token_sort_ratio",
    "address_ratio", "address_token_sort_ratio",
    "registration_id_exact", "country_exact", "city_exact", "postcode_exact",
    "registration_id_disagreement",
    "company_name_one_missing", "company_name_both_missing",
    "registration_id_one_missing", "registration_id_both_missing",
    "address_one_missing", "address_both_missing",
    "city_one_missing", "city_both_missing",
    "postcode_one_missing", "postcode_both_missing",
    "country_one_missing", "country_both_missing",
)
COMPARED_FIELDS = ("company_name", "registration_id", "address", "city", "postcode", "country")
NORMALIZED_FIELDS = tuple(f"{field}_normalized" for field in COMPARED_FIELDS)
FEATURE_VERSION = "1.0"


def _text(value: str | None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("Feature values must be text; never convert identifiers to numbers")
    return value.strip()


def build_pair_features(left: Mapping, right: Mapping) -> dict[str, float]:
    """Accept only six normalised observed values, with no IDs or label metadata.

    Missing/missing always gives zero similarity and zero exact agreement. Its
    explicit missingness indicator can be 1, which is not agreement evidence.
    """
    for record in (left, right):
        if set(record) != set(NORMALIZED_FIELDS):
            raise ValueError("Feature inputs must contain only the six allowed normalised observed fields")
    values: dict[str, float] = {}
    compared = {}
    for field in COMPARED_FIELDS:
        a, b = _text(left[f"{field}_normalized"]), _text(right[f"{field}_normalized"])
        compared[field] = (a, b)
        values[f"{field}_one_missing"] = float(bool(a) != bool(b))
        values[f"{field}_both_missing"] = float(not a and not b)
    for field in ("company_name", "address"):
        a, b = compared[field]
        values[f"{field}_ratio"] = fuzz.ratio(a, b, processor=None) / 100 if a and b else 0.0
        values[f"{field}_token_sort_ratio"] = fuzz.token_sort_ratio(a, b, processor=None) / 100 if a and b else 0.0
    for field in ("registration_id", "country", "city", "postcode"):
        a, b = compared[field]
        values[f"{field}_exact"] = float(bool(a) and bool(b) and a == b)
    a, b = compared["registration_id"]
    values["registration_id_disagreement"] = float(bool(a) and bool(b) and a != b)
    return {name: values[name] for name in FEATURE_NAMES}


def build_feature_frame(pairs: Sequence[Mapping], records: Mapping[str, Mapping]) -> pd.DataFrame:
    """Pair IDs locate records outside the matrix; pair metadata is never copied.

    Values in records must already be projected onto NORMALIZED_FIELDS. The
    caller separately validates split membership and candidate uniqueness.
    """
    rows = []
    for pair in pairs:
        left, right = pair["record_id_left"], pair["record_id_right"]
        if left == right:
            raise ValueError("Cannot build features for a self-pair")
        rows.append(build_pair_features(records[left], records[right]))
    frame = pd.DataFrame(rows, columns=list(FEATURE_NAMES), dtype=float)
    validate_feature_frame(frame)
    return frame


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Reject extra/reordered columns at both fitting and prediction boundaries."""
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != FEATURE_NAMES:
        raise ValueError("Model matrix must contain exactly the ordered feature allowlist")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("All model features must be numeric")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("Features must be finite values in [0, 1]")


def predict_rule_based(frame: pd.DataFrame) -> np.ndarray:
    """Fixed illustrative rules for this demo, not banking policy."""
    validate_feature_frame(frame)
    same_country = frame["country_exact"] == 1
    rule_a = same_country & (frame["registration_id_exact"] == 1)
    rule_b = (same_country & (frame["company_name_ratio"] >= 0.95)
              & (frame["address_ratio"] >= 0.90)
              & (frame["registration_id_disagreement"] == 0))
    return (rule_a | rule_b).to_numpy(dtype=int)
