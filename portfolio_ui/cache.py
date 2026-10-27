"""Cache only saved public synthetic artifacts and the read-only fitted model."""

from pathlib import Path

import streamlit as st

from portfolio_ui import artifacts as ad


@st.cache_data(show_spinner=False)
def catalog(root_string, stamp):
    return ad.load_catalog(Path(root_string))


@st.cache_data(show_spinner=False)
def samples(_catalog, fingerprint):
    return ad.load_samples(_catalog)


@st.cache_data(show_spinner=False)
def clusters(_catalog, fingerprint, split):
    return ad.load_cluster_bundle(_catalog, split)


@st.cache_data(show_spinner=False)
def hierarchy(_catalog, fingerprint):
    return ad.load_reference_hierarchy(_catalog)


@st.cache_data(show_spinner=False)
def evaluation_truth(_catalog, fingerprint):
    return ad.load_evaluation_truth(_catalog)


@st.cache_data(show_spinner=False)
def embedding_report(_catalog, fingerprint):
    return ad.read_json(_catalog.checked_path("reports/embedding_experiment/comparison.json"))


@st.cache_resource(show_spinner=False)
def model(_catalog, fingerprint):
    return ad.load_frozen_model(_catalog)


@st.cache_data(show_spinner=False)
def indian_demo(_catalog, fingerprint):
    from portfolio_ui.india import load_demo
    return load_demo(_catalog.root, _catalog)
