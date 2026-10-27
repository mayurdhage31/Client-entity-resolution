"""Stage 7 local portfolio UI. Importing this module does not launch the app."""

from pathlib import Path

import streamlit as st

from portfolio_ui import artifacts, cache, india
from portfolio_ui.screens import RENDERERS, SECTIONS


def main(root=None):
    st.set_page_config(page_title="Banking Client Entity Resolution", page_icon="🏦", layout="wide")
    st.title("Banking Client Entity Resolution")
    st.caption("Duplicate detection and corporate hierarchy exploration · Personal portfolio project")
    st.caption("INDIAN DEMO · Real company names; simulated records and errors. No actual customer data, affiliation or endorsement.")
    try:
        css_path = Path(__file__).resolve().parent / "portfolio_ui/styles.css"
        # Only a static, repository-owned stylesheet uses HTML rendering.
        st.markdown("<style>" + css_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
        root = Path(root) if root is not None else artifacts.PROJECT_ROOT
        stamp = artifacts.artifact_stamp(root)
        catalog = cache.catalog(str(root), stamp)
        indian_stamp = india.demo_stamp(root)
        fingerprint = (str(root.resolve()), stamp, indian_stamp)
        demo = cache.indian_demo(catalog, fingerprint)
        with st.sidebar:
            st.subheader("Indian finance demo")
            st.caption("Learn with simulated company records.")
            section = st.radio("Explore", SECTIONS, key="navigation")
            st.divider()
            st.caption("Indian demo is the default dataset. Original v1 benchmark results are separate.")
            st.caption("Edits stay in your session. Saved records and predictions remain unchanged.")
        RENDERERS[section](catalog, fingerprint, demo)
        st.divider()
        st.caption("Real public names, simulated records. Predictions are suggestions for review, not approved record merges.")
    except artifacts.ArtifactError as error:
        st.error(str(error))
        st.caption("Restore the required saved artifacts using the project copy described in README.md. "
                   "The app stops here and never regenerates data, trains a model or reruns evaluation.")
        st.stop()
    except (OSError, KeyError, TypeError, ValueError) as error:
        st.error("A required app file or saved artifact could not be read with the expected schema. "
                 "Check the pinned requirements and restore the original project artifacts.")
        st.text(f"Problem type: {type(error).__name__}")
        st.stop()


if __name__ == "__main__":
    main()
