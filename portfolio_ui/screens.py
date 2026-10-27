"""Beginner-friendly Indian demo screens; original benchmark stays separate."""

import pandas as pd
import streamlit as st

import prepare_candidates as pc
from portfolio_ui import cache, india, presentation as ui
from portfolio_ui.scoring import FIELDS, FIELD_LABELS, evidence_rows, score_records


SECTIONS = ("Start here", "Compare two company records", "Explore duplicate groups",
            "Explore corporate relationships", "How it works and limitations")


def intro(title, purpose, selection, interpretation):
    st.header(title)
    st.markdown("**What this page does:** " + purpose)
    st.markdown("**What to select:** " + selection)
    st.markdown("**How to interpret the result:** " + interpretation)


def _clear_result():
    st.session_state.pop("pair_result", None)


def _reset_copies():
    _clear_result()
    for side in ("left", "right"):
        for field in FIELDS:
            st.session_state.pop(side + "_" + field, None)


def _scenario_changed(demo):
    selected = st.session_state.get("guided_example")
    if selected:
        example = next(row for row in demo.examples if row["scenario_id"] == selected)
        st.session_state["record_a"] = example["record_id_left"]
        st.session_state["record_b"] = example["record_id_right"]
        st.session_state["comparison_mode"] = "Choose existing records"
    _reset_copies()


def _record_changed():
    st.session_state["guided_example"] = None
    _reset_copies()


def _try_example(demo):
    st.session_state["navigation"] = SECTIONS[1]
    st.session_state["guided_example"] = demo.examples[0]["scenario_id"]
    _scenario_changed(demo)


def _original_benchmark(catalog):
    metrics = catalog.report["metrics"]
    block, cluster = metrics["blocking"], metrics["clustering"]
    st.warning("Original v1 benchmark results — not Indian-demo performance")
    st.write(f"This older benchmark covers {block['record_count']} fictional v1 test records and "
             f"{block['candidate_pairs']} candidates. Its scores do not measure the new Indian examples.")
    st.dataframe(pd.DataFrame(ui.metric_comparison(catalog.report)), hide_index=True, width="stretch")
    st.caption("Candidate precision = TP/(TP+FP); candidate recall = TP/(TP+FN); candidate F1 = 2TP/(2TP+FP+FN). "
               f"Direct end-to-end recall uses all {block['all_true_duplicate_pairs']} true duplicate pairs, including blocking misses.")
    st.write(f"Cluster all-pairs F1: {ui.percent(cluster['pairwise_f1'])}, across {cluster['all_possible_pairs']:,} unordered v1 pairs.")
    method = st.selectbox("Original v1 confusion matrix", ("logistic_regression", "rule_based"),
                          format_func=lambda value: "Logistic regression" if value == "logistic_regression" else "Fixed rules", key="matrix_method")
    st.plotly_chart(ui.confusion_figure(metrics["candidate_classification"][method]), width="stretch", key="candidate_confusion")
    st.caption("Rows are actual evaluation labels; columns are model decisions. This matrix uses original v1 candidate pairs only.")


def start(catalog, fingerprint, demo):
    intro(SECTIONS[0], "See how two differently written company records can describe the same legal entity.",
          "Choose Try an example, then compare the two selected records.",
          "The model gives a suggestion, not proof of identity or permission to merge records.")
    st.subheader("One company, two ways of writing its name")
    example = demo.examples[0]
    left, right = demo.records[example["record_id_left"]], demo.records[example["record_id_right"]]
    columns = st.columns(2)
    for column, label, record in zip(columns, ("Record A", "Record B"), (left, right)):
        with column, st.container(border=True):
            st.markdown("**" + label + "**")
            st.text(record["company_name"])
            st.caption(record["source_system"])
    st.write("Would you recognise these as the same company? Compare their spelling, demo identifier and simulated address.")
    st.button("Try an example", type="primary", on_click=_try_example, args=(demo,), key="try_example")
    st.caption(f"Indian demo: {len(demo.records)} simulated source records using {len(demo.companies)} familiar public company names. "
               "No Indian-demo accuracy figure has been calculated. The unchanged model may make mistakes on these examples.")
    with st.expander("Original v1 benchmark details — separate from the Indian demo", expanded=False):
        _original_benchmark(catalog)


def _details_table(record):
    labels = {**FIELD_LABELS, "registration_id": "Fake demo identifier", "postcode": "Synthetic PIN code"}
    return pd.DataFrame([{"Field": labels[field], "Selected value": record[field] or "(missing)"} for field in FIELDS])


def compare(catalog, fingerprint, demo):
    intro(SECTIONS[1], "Compare two Indian demo records using the existing model.",
          "Pick Record A and Record B, or choose a guided example; then press Compare selected records.",
          "Read the actual model decision alongside the observed evidence. The score is uncalibrated, not confidence.")
    example_lookup = {row["scenario_id"]: row for row in demo.examples}
    st.selectbox("Guided example", [None, *example_lookup], key="guided_example",
                 format_func=lambda key: "Choose your own pair" if key is None else example_lookup[key]["label"],
                 on_change=_scenario_changed, args=(demo,))
    mode = st.radio("Comparison mode", ("Choose existing records", "Advanced: edit a copy"), horizontal=True,
                    key="comparison_mode", on_change=_reset_copies)
    default = demo.examples[0]
    options_a = india.record_options(demo)
    if st.session_state.get("record_a") not in options_a:
        st.session_state["record_a"] = default["record_id_left"]
    left_column, right_column = st.columns(2)
    selected_a = left_column.selectbox("Record A", options_a, key="record_a",
                                       format_func=lambda key: india.record_label(demo.records[key]), on_change=_record_changed)
    options_b = india.record_options(demo, exclude=selected_a)
    if st.session_state.get("record_b") not in options_b:
        st.session_state["record_b"] = default["record_id_right"] if default["record_id_right"] in options_b else options_b[0]
        _reset_copies()
    selected_b = right_column.selectbox("Record B", options_b, key="record_b",
                                        format_func=lambda key: india.record_label(demo.records[key]), on_change=_record_changed)
    saved_a, saved_b = demo.records[selected_a], demo.records[selected_b]
    left_column.dataframe(_details_table(saved_a), hide_index=True, width="stretch")
    right_column.dataframe(_details_table(saved_b), hide_index=True, width="stretch")
    left = {field: saved_a[field] for field in FIELDS}
    right = {field: saved_b[field] for field in FIELDS}
    with st.expander("Advanced: edit a copy", expanded=mode != "Choose existing records"):
        st.caption("Saved records are read-only. Choose Advanced: edit a copy above to edit temporary copies. "
                   "Edits stay in this session and never change data, groups or benchmark results.")
        if mode == "Advanced: edit a copy":
            for column, side, values in zip(st.columns(2), ("left", "right"), (left, right)):
                with column:
                    for field in FIELDS:
                        key = side + "_" + field
                        st.session_state.setdefault(key, values[field])
                        values[field] = st.text_input(FIELD_LABELS[field] + (" — A" if side == "left" else " — B"), key=key,
                                                      on_change=_clear_result, max_chars=1000)
    # A signature independently guards against stale results even if a widget is
    # changed programmatically without firing its callback.
    signature = (selected_a, selected_b, mode, tuple(left.items()), tuple(right.items()), fingerprint)
    if st.session_state.get("pair_result", {}).get("input_signature") != signature:
        _clear_result()
    selected_example = st.session_state.get("guided_example")
    if selected_example:
        example = example_lookup[selected_example]
        expected = india.educational_expectation(demo, example)
        st.info("Educational expectation from simulation labels: " + expected + ". This label is separate from the model's prediction.")
        if example["label"] == "Related but separate companies":
            st.caption("The supplied public reference describes a parent–subsidiary relationship. Related companies are still different legal entities.")
        if mode != "Choose existing records":
            st.caption("The educational expectation describes the selected saved pair. Your temporary edits can change its evidence.")
    if st.button("Compare selected records", type="primary", key="compare_selected"):
        try:
            pipeline, threshold = cache.model(catalog, fingerprint)
            result = score_records(pipeline, threshold, left, right)
            result["input_signature"] = signature
            st.session_state["pair_result"] = result
        except ValueError as error:
            _clear_result()
            st.error(str(error))
    result = st.session_state.get("pair_result")
    if result:
        st.subheader("Actual model prediction")
        (st.success if result["prediction"] else st.info)("Predicted match" if result["prediction"] else "Predicted non-match")
        a, b = st.columns(2)
        a.metric("Uncalibrated model score", f"{result['score']:.4f}")
        b.metric("Saved decision threshold", f"{result['threshold']:.2f}")
        st.caption("A score at or above the threshold predicts a match. It is not a calibrated probability or confidence estimate. "
                   "This direct comparison bypasses candidate blocking and does not change any saved predictions.")
        st.subheader("What the records show")
        evidence = pd.DataFrame(evidence_rows(result)).rename(columns={"Normalized A": "Cleaned value A", "Normalized B": "Cleaned value B"})
        st.dataframe(evidence, hide_index=True, width="stretch")
        st.caption("Missing values do not count as agreement. These comparisons describe observed fields; they do not prove legal identity or explain causality.")
    with st.expander("Technical record IDs and feature details", expanded=False):
        st.text("Record A: " + selected_a)
        st.text("Record B: " + selected_b)
        if result:
            st.dataframe(pd.DataFrame([{"Feature": key, "Value": value} for key, value in result["features"].items()]),
                         hide_index=True, width="stretch")


def groups(catalog, fingerprint, demo):
    intro(SECTIONS[2], "Explore the duplicate groups predicted from the Indian demo records.",
          "Filter by an observed company name or source system, then choose a group. Single-record groups are included.",
          "Filters find whole groups; they never remove members or repair the model's mistakes. A group is not an approved merge.")
    company_column, source_column, size_column = st.columns(3)
    company_names = sorted({row["company_name"] for row in demo.records.values()}, key=str.casefold)
    sources = sorted({row["source_system"] for row in demo.records.values()})
    company = company_column.selectbox("Company name as recorded", [None, *company_names], key="group_company",
                                       format_func=lambda value: "All observed names" if value is None else value or "(company name missing)")
    source = source_column.selectbox("Source system", [None, *sources], key="group_source",
                                     format_func=lambda value: "All demo systems" if value is None else value)
    size = size_column.selectbox("Group size", ("All groups", "Single-record groups", "Multiple-record groups"), key="group_size")
    review = st.checkbox("Only groups flagged for review", key="group_review_only")
    selected_ids = india.filtered_groups(demo, company, source, size, review)
    st.caption(f"Showing {len(selected_ids)} of {len(demo.bundle['by_cluster'])} saved predicted groups. "
               "A group appears when at least one member matches the filters; every member remains visible.")
    if not selected_ids:
        st.info("No groups match these filters. Choose All observed names, All demo systems, or clear the review filter.")
        return
    if st.session_state.get("selected_group") not in selected_ids:
        st.session_state["selected_group"] = selected_ids[0]
    group = st.selectbox("Predicted duplicate group", selected_ids, key="selected_group", format_func=lambda key: india.group_label(demo, key))
    members = demo.bundle["by_cluster"][group]
    edges = [row for row in demo.bundle["edges"] if row["cluster_id"] == group]
    flags = [row for row in demo.bundle["flags"] if row["cluster_id"] == group]
    summary = next(row for row in demo.bundle["summaries"] if row["cluster_id"] == group)
    first, second, third = st.columns(3)
    first.metric("Records in this group", len(members))
    second.metric("Direct model links", len(edges))
    third.metric("Extra pairs connected by a path", summary["inferred_without_direct_edge_count"])
    if flags:
        st.warning("Review flag: this group contains different nonempty demo identifiers for the same country. "
                   "The flag asks for human review; it does not change membership.")
    shown = [{"Record": row["record_id"][-8:], "Company name": row["company_name"] or "(missing)", "Source system": row["source_system"],
              "Demo identifier": row["registration_id"], "Simulated address": row["address"], "City": row["city"],
              "Synthetic PIN": row["postcode"], "Country": row["country"]} for row in members]
    st.dataframe(pd.DataFrame(shown), hide_index=True, width="stretch")
    st.download_button("Download this group's records", ui.download_records(members),
                       file_name="india_demo_group_" + group[-8:] + ".csv", mime="text/csv", key="download_cluster")
    st.plotly_chart(ui.duplicate_figure(members, edges, bool(flags), compact_ids=True), width="stretch", key="duplicate_graph")
    st.caption("Lines are saved direct model matches. If A links to B and B links to C, all three share a group even without an A–C line. "
               "A single incorrect line can therefore join different companies. A single-record group has no direct lines.")
    with st.expander("Full technical IDs, direct links and review evidence", expanded=False):
        st.text("Predicted group ID: " + group)
        st.dataframe(pd.DataFrame(members)[list(pc.OBSERVED_FIELDS)], hide_index=True, width="stretch")
        if edges:
            st.dataframe(pd.DataFrame(edges), hide_index=True, width="stretch")
        if flags:
            st.dataframe(pd.DataFrame(flags), hide_index=True, width="stretch")
    with st.expander("Educational labels only — not predicted company identities", expanded=False):
        st.caption("Simulation truth can check this group, but is never used to build or repair it or to link it to public corporate relationships.")
        if st.checkbox("Show separate simulation labels", key="show_cluster_truth"):
            path = india.checked_data_path(demo, "data/india_demo/ground_truth/record_entity_map.csv")
            truth = {row["record_id"]: row["true_entity_id"] for row in pc.read_csv(path, pc.TRUTH_FIELDS)}
            st.dataframe(pd.DataFrame([{"Record ID": row["record_id"], "Simulation entity label (not a prediction)": truth[row["record_id"]]}
                                       for row in members]), hide_index=True, width="stretch")


def relationships(catalog, fingerprint, demo):
    intro(SECTIONS[3], "Look at an officially sourced relationship between separate Indian companies.",
          "Choose a family from the supplied reference list, and follow the parent-to-child arrow.",
          "This is public reference information, not a model prediction or a complete ownership structure.")
    st.info("Supplied public reference information — not predictions; not an exhaustive ownership structure")
    graph, names = india.reference_graph(demo)
    families = ui.reference_families(graph)
    def family_label(index):
        subgraph = graph.subgraph(families[index])
        roots = sorted(node for node in subgraph if subgraph.in_degree(node) == 0)
        return " / ".join(names[node] for node in roots) + f" | {len(subgraph)} separate companies"
    selected = st.selectbox("Supplied corporate family", range(len(families)), key="hierarchy_family", format_func=family_label)
    family = graph.subgraph(families[selected]).copy()
    st.plotly_chart(ui.hierarchy_figure(family, names), width="stretch", key="reference_graph")
    for row in demo.relationships:
        if row["parent_entity_id"] in family:
            st.write(names[row["parent_entity_id"]] + " → " + names[row["child_entity_id"]])
            st.caption("Relationship: parent to subsidiary. Source: " + row["relationship_source"] + ". Retrieved: " + row["retrieved_on"] + ".")
            st.link_button("Read the official relationship source", row["source_url"])
    st.caption("Shared names alone do not establish ownership. No predicted duplicate group is automatically linked to this public reference view.")


def method(catalog, fingerprint, demo):
    intro(SECTIONS[4], "Understand the demonstration and where it can make mistakes.",
          "Read the short steps; expand technical details or open an official name source if useful.",
          "Real names make the examples familiar, but simulated records and a model suggestion are not verified bank data.")
    st.write(demo.info["disclaimer"])
    steps = [
        ("1. Create teaching records", "Six verified public names are written in different ways across four fictional DEMO systems. "
         "Addresses, PIN codes, DEMO-IN identifiers, typos and missing details are simulated, not copied bank or customer records."),
        ("2. Find possible duplicates", "The existing rules select possible pairs using nonempty registration/country, postcode/name "
         "or city/name clues. Fuzzy name or address retrieval can find more possibilities. Missing values are not agreements."),
        ("3. Compare the observed fields", "The same cleaning functions and 21 field-comparison features feed the already trained model. "
         "The Indian demo does not retrain the model, change the saved threshold or tune the candidate rules."),
        ("4. Suggest duplicate groups", "Predicted direct matches become graph lines. Connected records share a group, including through paths. "
         "Every record stays visible, even if it has no matches. Review flags do not repair groups."),
        ("5. Keep public relationships separate", "The supplied Bajaj parent–subsidiary reference has a source and retrieval date. "
         "Related companies remain separate legal entities; the model did not infer this relationship."),
    ]
    for title, text in steps:
        st.markdown("**" + title + "**")
        st.write(text)
    st.subheader("Important limitations")
    st.markdown("- No accuracy figure has been calculated for this Indian demo. Original v1 benchmark scores belong to the older fictional dataset.\n"
                "- The frozen model was trained on the original synthetic data; Indian abbreviations and sparse records can produce wrong decisions.\n"
                "- Scores are uncalibrated model scores, not confidence or verified identity.\n"
                "- Missed candidates cannot become direct model matches. One wrong graph link can join entire groups.\n"
                "- Real company names do not imply affiliation, endorsement or access to real customer information.\n"
                "- Source pages can change. The supplied relationship is a retrieved snapshot, not an exhaustive or real-time ownership register.")
    with st.expander("Official name verification sources", expanded=False):
        for company in demo.companies:
            st.link_button(company["company_name"], company["source_url"])
            st.caption(company["verification_note"] + " Retrieved " + company["retrieved_on"] + ".")
    with st.expander("Technical pipeline and saved embedding experiment", expanded=False):
        st.write("The original training/validation/test split kept entities and complete corporate families apart before creating pairs. "
                 "StandardScaler and logistic regression fitted only training candidates. The saved Indian batch uses that same model without fitting.")
        st.caption("Exhaustive fuzzy comparisons are fine for this small demo; large datasets need a scalable retrieval index.")
        experiment = cache.embedding_report(catalog, fingerprint)
        st.dataframe(pd.DataFrame([{"Model": label, "Original validation F1": ui.percent(experiment["comparison"][key]["f1"])}
                                   for key, label in (("stage3_logistic_regression", "Original 21 features"),
                                                      ("embedding_experiment", "With name/address embeddings"))]), hide_index=True, width="stretch")
        st.write("No validation F1 improvement was measured, so the simpler original model was retained. "
                 "The app reads this saved report without loading torch, an encoder or downloading weights.")
        st.text("Frozen model SHA-256: " + demo.run["model_sha256"])
        st.text("Saved threshold: " + str(demo.run["threshold"]))
        st.caption(demo.run["performance"])


RENDERERS = dict(zip(SECTIONS, (start, compare, groups, relationships, method)))
