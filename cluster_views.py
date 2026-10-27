"""Two offline Plotly views with separate prediction and reference data flows."""

from __future__ import annotations

import csv
import html
import math
from itertools import combinations
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

import prepare_candidates as pc


HIERARCHY_LABEL = "Supplied synthetic relationships; not inferred by the matching model"


def safe_text(value):
    return html.escape(str(value), quote=True)


def html_table(headers, rows):
    header = "".join(f"<th>{safe_text(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{safe_text(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def write_view(path, title, paragraphs, figure, tables):
    """Inline Plotly JS and data; no CDN, external CSS, images, or MathJax."""
    plot = figure.to_html(full_html=False, include_plotlyjs=True, include_mathjax=False,
                          div_id="local_network_view", config={"displaylogo": False, "scrollZoom": True})
    intro = "".join(f"<p>{safe_text(paragraph)}</p>" for paragraph in paragraphs)
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{safe_text(title)}</title>"
        "<style>body{font:15px Arial,sans-serif;max-width:1250px;margin:24px auto;padding:0 18px;color:#172b4d}"
        "table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}"
        "td,th{border:1px solid #ccd4df;padding:8px;text-align:left;overflow-wrap:anywhere}"
        "th{background:#eef3f9}p{line-height:1.5}</style></head>"
        f"<body><h1>{safe_text(title)}</h1>{intro}{plot}{''.join(tables)}</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def choose_validation_example(graph, maximum_records=8):
    """Select by predicted graph structure only, never by evaluation outcomes."""
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    eligible = [members for members in components if 2 <= len(members) <= maximum_records]
    if eligible:
        eligible.sort(key=lambda members: (
            0 if math.comb(len(members), 2) > graph.subgraph(members).number_of_edges() else 1,
            -len(members), members,
        ))
        return eligible[0], len(eligible[0])
    if any(len(members) > 1 for members in components):
        members = min((members for members in components if len(members) > 1), key=lambda values: (len(values), values))
        displayed = list(nx.bfs_tree(graph, members[0], sort_neighbors=sorted))[:maximum_records]
        return tuple(sorted(displayed)), len(members)
    return tuple(sorted(graph)[:maximum_records]), 1 if graph else 0


def render_duplicate_example(graph, records, destination):
    displayed, full_size = choose_validation_example(graph)
    view = graph.subgraph(displayed).copy()
    positions = nx.spring_layout(view, seed=42)
    figure = go.Figure()
    for index, (left, right) in enumerate(sorted(tuple(sorted(edge)) for edge in view.edges)):
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        edge = view.edges[left, right]
        figure.add_trace(go.Scatter(x=[float(x0), float(x1)], y=[float(y0), float(y1)], mode="lines",
                                    line={"width": 2.5, "color": "#2563eb"}, name="Direct predicted edge",
                                    showlegend=index == 0, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=[float((x0 + x1) / 2)], y=[float((y0 + y1) / 2)], mode="markers",
                                    marker={"size": 14, "color": "rgba(0,0,0,0)"}, showlegend=False,
                                    text=[f"Direct model score: {edge['model_score']:.6f}<br>"
                                          f"Methods: {safe_text(edge['blocking_rules'])}"], hovertemplate="%{text}<extra></extra>"))
    labels, hover = [], []
    for record_id in displayed:
        record = records[record_id]
        name = record["company_name"] or "(company name missing)"
        labels.append(safe_text(name) + "<br>" + safe_text(record_id))
        hover.append("<br>".join(safe_text(value) for value in (
            name, record_id, "Source: " + record["source_system"], "Address: " + record["address"],
            "Registration: " + record["registration_id"], "Country: " + record["country"],
        )))
    figure.add_trace(go.Scatter(x=[float(positions[node][0]) for node in displayed],
                                y=[float(positions[node][1]) for node in displayed], mode="markers+text",
                                marker={"size": 19, "color": "#0f766e", "line": {"width": 1, "color": "white"}},
                                text=labels, textposition="top center", hovertext=hover,
                                hovertemplate="%{hovertext}<extra></extra>", showlegend=False, cliponaxis=False))
    figure.update_layout(height=650, template="plotly_white", dragmode="pan", hovermode="closest",
                         margin={"l": 80, "r": 80, "t": 100, "b": 60},
                         xaxis={"visible": False}, yaxis={"visible": False})
    relations = []
    for left, right in combinations(displayed, 2):
        if graph.has_edge(left, right):
            relation, score = "Direct model match edge", f"{graph.edges[left, right]['model_score']:.6f}"
        elif nx.has_path(graph, left, right):
            relation, score = "Inferred same-component membership only", ""
        else:
            relation, score = "Different predicted components", ""
        relations.append((left, right, relation, score))
    paragraphs = [
        "SYNTHETIC DATA ONLY. Validation prediction example selected using graph structure, without truth labels.",
        "Lines are direct model matches at the saved threshold. Inferred same-component membership is listed separately in the table.",
        "A-B and B-C edges put A, B, and C in one connected component even when A-C has no direct match edge.",
        "A predicted cluster is not an approved record merge. Registration review flags never change membership.",
        f"Showing {len(displayed)} records; selected component size {full_size}. Full outputs retain every record and component.",
        "Hover nodes for observed fields and edge midpoints for uncalibrated model scores/provenance. Drag to pan; use the toolbar to zoom.",
    ]
    tables = [html_table(("Observed company name", "Record ID", "Source", "Observed registration", "Observed country"),
                         [(records[node]["company_name"], node, records[node]["source_system"],
                           records[node]["registration_id"], records[node]["country"]) for node in displayed]),
              html_table(("Record A", "Record B", "Relationship in this prediction", "Direct model score"), relations)]
    write_view(destination, "Validation duplicate-cluster example: predicted membership", paragraphs, figure, tables)
    return {"file": destination.name, "displayed_record_ids": list(displayed), "full_component_size": full_size,
            "displayed_direct_edges": view.number_of_edges(), "uses_truth_labels": False}


def build_reference_graph(relationships):
    pc.validate_columns(relationships, pc.PARENT_FIELDS, "Supplied hierarchy")
    graph = nx.DiGraph()
    for row in relationships:
        parent, child = row["parent_entity_id"], row["child_entity_id"]
        pc.require(parent and child and parent != child, "Invalid supplied parent/child IDs")
        pc.require(not graph.has_edge(parent, child), "Duplicate supplied relationship")
        graph.add_edge(parent, child, relationship_source=row["relationship_source"])
    pc.require(nx.is_directed_acyclic_graph(graph), "Supplied hierarchy contains a cycle")
    return graph


def read_canonical_display_names(path, reference_nodes):
    """Only read the ID/name projection as display labels for supplied endpoints."""
    names = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        pc.require({"true_entity_id", "company_name"} <= set(reader.fieldnames or []), "Missing reference display-label columns")
        for row in reader:
            entity_id = row["true_entity_id"]
            if entity_id in reference_nodes:
                pc.require(entity_id not in names, "Duplicate canonical display-label ID")
                names[entity_id] = row["company_name"] or entity_id
    return {node: names.get(node, node) for node in reference_nodes}


def hierarchy_positions(graph):
    positions = {}
    components = sorted(tuple(sorted(values)) for values in nx.weakly_connected_components(graph))
    for index, component in enumerate(components):
        base_x, base_y = (index % 3) * 8.0, -(index // 3) * 6.0
        for level, nodes in enumerate(nx.topological_generations(graph.subgraph(component))):
            ordered = sorted(nodes)
            for offset, node in enumerate(ordered):
                positions[node] = (base_x + (offset - (len(ordered) - 1) / 2) * 1.9, base_y - level * 1.7)
    return positions


def render_reference_hierarchy(relationships_path, names_path, destination):
    relationships = pc.read_csv(relationships_path, pc.PARENT_FIELDS)
    graph = build_reference_graph(relationships)  # Nodes and edges come ONLY from the reference file.
    names = read_canonical_display_names(names_path, set(graph))
    positions = hierarchy_positions(graph)
    figure = go.Figure()
    for parent, child in sorted(graph.edges):
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        figure.add_annotation(x=x1 - (x1 - x0) * .10, y=y1 - (y1 - y0) * .10,
                              ax=x0 + (x1 - x0) * .10, ay=y0 + (y1 - y0) * .10,
                              xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                              arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#475569")
    ordered = sorted(graph)
    figure.add_trace(go.Scatter(x=[positions[node][0] for node in ordered], y=[positions[node][1] for node in ordered],
                                mode="markers+text", text=[safe_text(node) for node in ordered], textposition="bottom center",
                                marker={"size": 17, "color": "#7c3aed"},
                                hovertext=[safe_text(names[node]) + "<br>" + safe_text(node) for node in ordered],
                                hovertemplate="%{hovertext}<extra></extra>", showlegend=False))
    rows = max(1, math.ceil(nx.number_weakly_connected_components(graph) / 3))
    figure.update_layout(height=max(500, rows * 260), template="plotly_white", dragmode="pan",
                         margin={"l": 40, "r": 40, "t": 60, "b": 60},
                         xaxis={"visible": False}, yaxis={"visible": False})
    table = html_table(("Parent entity ID", "Canonical parent display name", "Child entity ID", "Canonical child display name", "Supplied source"),
                       [(row["parent_entity_id"], names[row["parent_entity_id"]], row["child_entity_id"],
                         names[row["child_entity_id"]], row["relationship_source"]) for row in relationships])
    write_view(destination, HIERARCHY_LABEL, [
        "SYNTHETIC REFERENCE DATA ONLY. Every arrow points from a supplied parent to its supplied child.",
        "Only reference-file endpoints are drawn. Canonical company names serve solely as display labels; hover nodes to see them.",
        "This view uses no record-to-entity truth mapping, predicted cluster membership, or matching-model outputs.",
        "Related companies remain separate legal entities. No linking of predicted record clusters to this hierarchy is asserted.",
    ], figure, [table])
    return {"file": destination.name, "label": HIERARCHY_LABEL, "entity_nodes": graph.number_of_nodes(),
            "supplied_relationship_edges": graph.number_of_edges(), "direction": "parent_to_child",
            "uses_record_truth_mapping": False, "uses_predicted_clusters": False}
