"""Safe display helpers. Saved graph edges and supplied reference edges stay separate."""

import csv
import html
import io
import textwrap

import networkx as nx
import plotly.graph_objects as go

import prepare_candidates as pc


TEAL = "#0f766e"
NAVY = "#18344c"
AMBER = "#b7791f"


def percent(value):
    return "Undefined" if value is None else f"{100 * value:.2f}%"


def safe(value):
    return html.escape(str(value), quote=True)


def wrapped(value, width=27):
    return "<br>".join(safe(part) for part in textwrap.wrap(str(value), width=width)) or "(name missing)"


def metric_comparison(report):
    metrics = report["metrics"]
    return [{"Method": label, "Candidate precision": percent(row["precision"]),
             "Candidate recall": percent(row["recall"]), "Candidate F1": percent(row["f1"]),
             "End-to-end direct recall": percent(metrics["end_to_end_direct_pair"][method]["end_to_end_direct_pair_recall"]),
             "TP": row["tp"], "FP": row["fp"], "TN": row["tn"], "FN": row["fn"]}
            for method, label in (("rule_based", "Fixed rules"), ("logistic_regression", "Logistic regression"))
            for row in [metrics["candidate_classification"][method]]]


def confusion_figure(row):
    values = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    labels = [[f"TN · {row['tn']}", f"FP · {row['fp']}"], [f"FN · {row['fn']}", f"TP · {row['tp']}"]]
    figure = go.Figure(go.Heatmap(
        z=values, x=["Predicted non-match", "Predicted match"], y=["Actual non-match", "Actual match"],
        text=labels, texttemplate="%{text}", textfont={"size": 18},
        colorscale=[[0, "#eff6f6"], [1, TEAL]], showscale=False,
        hovertemplate="%{y}<br>%{x}<br>Pairs: %{z}<extra></extra>", xgap=5, ygap=5,
    ))
    figure.update_layout(height=350, margin={"l": 10, "r": 10, "t": 20, "b": 20},
                         xaxis_title="Fixed prediction", yaxis_title="Evaluation truth", template="plotly_white")
    figure.update_yaxes(autorange="reversed")
    return figure


def filtered_cluster_ids(bundle, query="", size_filter="All clusters", flagged_only=False):
    query = query.strip().casefold()
    flagged = {row["cluster_id"] for row in bundle["flags"]}
    selected = []
    for row in bundle["summaries"]:
        cluster_id, count = row["cluster_id"], int(row["record_count"])
        if size_filter == "Singletons only" and count != 1:
            continue
        if size_filter == "Multiple records" and count <= 1:
            continue
        if flagged_only and cluster_id not in flagged:
            continue
        values = [cluster_id, *(str(member[field]) for member in bundle["by_cluster"][cluster_id] for field in pc.OBSERVED_FIELDS)]
        if query and not any(query in value.casefold() for value in values):
            continue
        selected.append(cluster_id)
    # Larger examples first, with a deterministic ID tie break. Never use truth.
    return sorted(selected, key=lambda cluster_id: (-len(bundle["by_cluster"][cluster_id]), cluster_id))


def download_records(members):
    """Original observed fields only; no reference labels or inferred identities."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=pc.OBSERVED_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row[field] for field in pc.OBSERVED_FIELDS} for row in members)
    return stream.getvalue().encode("utf-8-sig")


def graph_layout(figure, height=360):
    figure.update_layout(template="plotly_white", height=height,
                         margin={"l": 35, "r": 35, "t": 35, "b": 30},
                         xaxis={"visible": False}, yaxis={"visible": False},
                         hovermode="closest", dragmode="pan", showlegend=False)
    return figure


def duplicate_figure(members, edges, flagged=False, compact_ids=False):
    """Draw only saved direct edges; do not infer additional match edges."""
    records = {row["record_id"]: row for row in members}
    graph = nx.Graph()
    graph.add_nodes_from(sorted(records))
    graph.add_edges_from((row["record_id_left"], row["record_id_right"]) for row in edges)
    if set(graph) != set(records):
        raise ValueError("A saved direct edge leaves the selected cluster.")
    positions = nx.spring_layout(graph, seed=42, scale=1.0)
    figure = go.Figure()
    hover_x, hover_y, hover_text = [], [], []
    for edge in edges:
        x0, y0 = positions[edge["record_id_left"]]
        x1, y1 = positions[edge["record_id_right"]]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                   line={"width": 2.5, "color": "#7ca8b5"}, hoverinfo="skip"))
        hover_x.append((x0 + x1) / 2)
        hover_y.append((y0 + y1) / 2)
        hover_text.append(f"Direct predicted edge<br>Uncalibrated score: {float(edge['model_score']):.4f}<br>"
                          f"Blocking: {safe(edge['blocking_rules'])}")
    figure.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker={"size": 18, "opacity": 0},
                               hovertext=hover_text, hovertemplate="%{hovertext}<extra></extra>"))
    ordered = sorted(records)
    figure.add_trace(go.Scatter(
        x=[positions[node][0] for node in ordered], y=[positions[node][1] for node in ordered],
        mode="markers+text", text=[wrapped(records[node]["company_name"]) for node in ordered], textposition="top center",
        textfont={"size": 12, "color": NAVY}, marker={"size": 22, "color": AMBER if flagged else TEAL,
                                                      "line": {"color": "white", "width": 2}},
        hovertext=[f"{safe(records[node]['company_name'] or '(name missing)')}<br>{safe(node[-8:] if compact_ids else node)}<br>"
                   f"Source: {safe(records[node]['source_system'])}<br>"
                   f"Registration: {safe(records[node]['registration_id'] or '(missing)')}<br>"
                   f"Address: {safe(records[node]['address'] or '(missing)')}" for node in ordered],
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    figure.update_xaxes(range=[-1.7, 1.7])
    figure.update_yaxes(range=[-1.5, 1.7])
    return graph_layout(figure)


def reference_families(graph):
    return sorted(tuple(sorted(component)) for component in nx.weakly_connected_components(graph))


def hierarchy_figure(graph, names):
    """The graph must come solely from supplied parent-to-child relationships."""
    positions = {}
    for depth, nodes in enumerate(nx.topological_generations(graph)):
        nodes = sorted(nodes)
        for offset, node in enumerate(nodes):
            positions[node] = ((offset - (len(nodes) - 1) / 2) * 2.7, -depth * 2.0)
    figure = go.Figure()
    for parent, child, data in sorted(graph.edges(data=True)):
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        figure.add_annotation(x=x1 - (x1 - x0) * .12, y=y1 - (y1 - y0) * .12,
                              ax=x0 + (x1 - x0) * .12, ay=y0 + (y1 - y0) * .12,
                              xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                              arrowhead=2, arrowsize=1.4, arrowwidth=2, arrowcolor="#78879d")
        figure.add_trace(go.Scatter(x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers",
                                   marker={"size": 18, "opacity": 0},
                                   hovertext=[f"Parent: {safe(names[parent])}<br>Child: {safe(names[child])}<br>"
                                              f"Source: {safe(data['relationship_source'])}"],
                                   hovertemplate="%{hovertext}<extra></extra>"))
    ordered = sorted(graph)
    figure.add_trace(go.Scatter(x=[positions[node][0] for node in ordered], y=[positions[node][1] for node in ordered],
                               mode="markers+text", marker={"size": 24, "color": "#5967a6", "line": {"color": "white", "width": 2}},
                               text=[wrapped(names[node]) for node in ordered], textposition="bottom center",
                               textfont={"size": 12, "color": NAVY},
                               hovertext=[f"{safe(names[node])}<br>Supplied entity: {safe(node)}" for node in ordered],
                               hovertemplate="%{hovertext}<extra></extra>"))
    if positions:
        xs, ys = zip(*positions.values())
        figure.update_xaxes(range=[min(xs) - 2, max(xs) + 2])
        figure.update_yaxes(range=[min(ys) - 1.2, max(ys) + 1])
    return graph_layout(figure, height=360)
