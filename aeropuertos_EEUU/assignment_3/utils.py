from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from bokeh.layouts import column
from bokeh.models import (
    Arrow,
    BasicTicker,
    Circle,
    ColumnDataSource,
    ColorBar,
    CustomJS,
    HoverTool,
    LinearColorMapper,
    MultiLine,
    NodesAndLinkedEdges,
    OpenHead,
    Scatter,
    WMTSTileSource,
)
from bokeh.palettes import Turbo256
from bokeh.plotting import figure, from_networkx


@dataclass
class AirportDatasetConfig:
    """Configurable dataset schema so new airport datasets can be plugged in."""

    nodes_path: str
    edges_path: str
    node_id_col: str = "node_id"
    node_name_col: str = "name"
    node_lat_col: str = "latitude"
    node_lon_col: str = "longitude"
    node_pop_col: str = "metro_pop"
    edge_source_col: str = "FromNodeId"
    edge_target_col: str = "ToNodeId"
    edge_weight_col: str = "Weight"
    edges_comment_prefix: str = "#"


def load_airport_data(config: AirportDatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load airport nodes and edge list with configurable schema."""
    nodes_df = pd.read_csv(config.nodes_path)

    edges_path = Path(config.edges_path)
    if edges_path.suffix.lower() == ".csv":
        edges_df = pd.read_csv(config.edges_path)
    else:
        with open(config.edges_path, "r", encoding="utf-8") as f:
            rows = [line.strip().split() for line in f if line.strip() and not line.startswith(config.edges_comment_prefix)]
        edges_df = pd.DataFrame(
            rows,
            columns=[config.edge_source_col, config.edge_target_col, config.edge_weight_col],
        )

    edges_df = edges_df.astype(
        {
            config.edge_source_col: int,
            config.edge_target_col: int,
            config.edge_weight_col: float,
        }
    )
    return nodes_df, edges_df


def build_directed_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    config: AirportDatasetConfig,
) -> nx.DiGraph:
    """Build a directed graph with airport attributes."""
    graph = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        node_id = int(row[config.node_id_col])
        graph.add_node(
            node_id,
            name=str(row[config.node_name_col]),
            pop=float(row[config.node_pop_col]),
            latitude=float(row[config.node_lat_col]),
            longitude=float(row[config.node_lon_col]),
            pos=(float(row[config.node_lon_col]), float(row[config.node_lat_col])),
        )

    for _, row in edges_df.iterrows():
        graph.add_edge(
            int(row[config.edge_source_col]),
            int(row[config.edge_target_col]),
            weight=float(row[config.edge_weight_col]),
        )

    return graph


def split_one_way_and_two_way(graph: nx.DiGraph) -> Tuple[nx.DiGraph, nx.Graph]:
    """Create one-way directed graph and two-way undirected graph."""
    one_way_edges = [(u, v) for u, v in graph.edges() if not graph.has_edge(v, u)]
    two_way_edges = [(u, v) for u, v in graph.edges() if graph.has_edge(v, u) and u < v]

    g_one_way = nx.DiGraph()
    g_one_way.add_nodes_from(graph.nodes(data=True))
    g_one_way.add_edges_from(one_way_edges)

    g_undirected = nx.Graph()
    g_undirected.add_nodes_from(graph.nodes(data=True))
    g_undirected.add_edges_from(two_way_edges)

    return g_one_way, g_undirected


def directed_metrics(graph: nx.DiGraph) -> Dict[str, float]:
    scc = list(nx.strongly_connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "scc_count": len(scc),
        "largest_scc_size": max((len(c) for c in scc), default=0),
    }


def undirected_metrics(graph: nx.Graph) -> Dict[str, float]:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    largest_cc = max(components, key=len) if components else set()
    g_lcc = graph.subgraph(largest_cc).copy() if largest_cc else nx.Graph()

    out = {
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": (2 * n_edges / n_nodes) if n_nodes else 0.0,
        "density": nx.density(graph) if n_nodes else 0.0,
        "connected_components": len(components),
        "largest_cc_size": len(largest_cc),
        "avg_clustering": nx.average_clustering(graph) if n_nodes else 0.0,
        "transitivity": nx.transitivity(graph) if n_nodes else 0.0,
    }

    if g_lcc.number_of_nodes() > 1:
        out["mean_shortest_path"] = nx.average_shortest_path_length(g_lcc)
        out["diameter"] = nx.diameter(g_lcc)
        out["radius"] = nx.radius(g_lcc)
    else:
        out["mean_shortest_path"] = np.nan
        out["diameter"] = np.nan
        out["radius"] = np.nan

    return out


def compute_degree_differences(graph: nx.DiGraph) -> Dict[int, int]:
    return {n: graph.in_degree(n) - graph.out_degree(n) for n in graph.nodes()}


def compute_centralities(graph: nx.Graph) -> Dict[str, Dict[int, float]]:
    closeness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    try:
        eigenvector = nx.eigenvector_centrality_numpy(graph)
    except Exception:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)

    try:
        katz = nx.katz_centrality_numpy(graph)
    except Exception:
        katz = nx.katz_centrality(graph, max_iter=1000)

    return {
        "Closeness Centrality": closeness,
        "Betweenness Centrality": betweenness,
        "Eigenvector Centrality": eigenvector,
        "Katz Centrality": katz,
    }


def build_summary_tables(graph_d: nx.DiGraph, graph_u: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d_df = pd.DataFrame([directed_metrics(graph_d)])
    u_df = pd.DataFrame([undirected_metrics(graph_u)])
    return d_df, u_df


def _subsample_graph_edges(graph: nx.Graph, max_edges: int, seed: int = 42) -> nx.Graph:
    if graph.number_of_edges() <= max_edges:
        return graph.copy()

    rng = random.Random(seed)
    sampled_edges = rng.sample(list(graph.edges()), max_edges)
    sampled = graph.__class__()
    sampled.add_nodes_from(graph.nodes(data=True))
    for u, v in sampled_edges:
        sampled.add_edge(u, v, **graph.get_edge_data(u, v))
    return sampled


def _to_web_mercator(lon: float, lat: float) -> Tuple[float, float]:
    lon = float(lon)
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    x = lon * 20037508.34 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * 6378137.0
    return x, y


def _positions(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    return {n: _to_web_mercator(graph.nodes[n]["longitude"], graph.nodes[n]["latitude"]) for n in graph.nodes()}


def _base_figure(title: str, pos: Dict[int, Tuple[float, float]], tooltips: List[Tuple[str, str]]):
    x_vals = [xy[0] for xy in pos.values()]
    y_vals = [xy[1] for xy in pos.values()]
    pad_x = max(1000000, (max(x_vals) - min(x_vals)) * 0.08)
    pad_y = max(1000000, (max(y_vals) - min(y_vals)) * 0.08)

    plot = figure(
        title=title,
        tooltips=tooltips,
        tools="pan,wheel_zoom,box_zoom,save,reset,tap",
        active_scroll="wheel_zoom",
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_range=(min(x_vals) - pad_x, max(x_vals) + pad_x),
        y_range=(min(y_vals) - pad_y, max(y_vals) + pad_y),
        width=950,
        height=600,
    )
    tile_source = WMTSTileSource(
        url="https://tiles.basemaps.cartocdn.com/light_all/{Z}/{X}/{Y}.png",
        attribution="&copy; OpenStreetMap contributors &copy; CARTO",
    )
    plot.add_tile(tile_source)
    plot.xaxis.axis_label = "Longitude"
    plot.yaxis.axis_label = "Latitude"
    return plot


def _attach_zoom_scaling(plot, source, base_size: int, min_size: int = 8, max_size: int = 26):
    initial_x_span = float(plot.x_range.end - plot.x_range.start)
    initial_y_span = float(plot.y_range.end - plot.y_range.start)

    callback = CustomJS(
        args=dict(source=source, x_range=plot.x_range, y_range=plot.y_range, initial_x_span=initial_x_span, initial_y_span=initial_y_span, base_size=base_size, min_size=min_size, max_size=max_size),
        code="""
            const currentXSpan = Math.max(1, x_range.end - x_range.start)
            const currentYSpan = Math.max(1, y_range.end - y_range.start)
            const zoomScale = Math.min(initial_x_span / currentXSpan, initial_y_span / currentYSpan)
            const sizes = source.data.size_base.map((value) => {
                return Math.max(min_size, Math.min(max_size, value * zoomScale))
            })
            source.data.size = sizes
            source.change.emit()
        """,
    )
    plot.x_range.js_on_change("start", callback)
    plot.x_range.js_on_change("end", callback)
    plot.y_range.js_on_change("start", callback)
    plot.y_range.js_on_change("end", callback)
    callback.code = callback.code


def _prepare_node_attributes(
    graph: nx.Graph,
    source_graph: nx.Graph,
    values: Dict[int, float] | None = None,
    roles: Dict[int, str] | None = None,
    value_field_name: str = "value",
):
    for n in graph.nodes():
        graph.nodes[n].update(source_graph.nodes[n])
        graph.nodes[n]["degree"] = int(source_graph.degree(n))
        if source_graph.is_directed():
            graph.nodes[n]["in_degree"] = int(source_graph.in_degree(n))
            graph.nodes[n]["out_degree"] = int(source_graph.out_degree(n))
        else:
            graph.nodes[n]["in_degree"] = int(source_graph.degree(n))
            graph.nodes[n]["out_degree"] = int(source_graph.degree(n))
        graph.nodes[n]["value"] = float(values[n]) if values is not None else 0.0
        if values is not None:
            graph.nodes[n][value_field_name] = float(values[n])
        if roles is not None:
            graph.nodes[n]["role"] = roles.get(n, "other")
        graph.nodes[n]["size_base"] = float(graph.nodes[n].get("size_base", 1.0))


def _node_tooltips(graph: nx.Graph, show_value: bool = False, show_role: bool = False):
    tooltips = [("Airport", "@name")]
    if graph.is_directed():
        tooltips.extend(
            [
                ("Degree", "@degree"),
                ("In Degree", "@in_degree"),
                ("Out Degree", "@out_degree"),
            ]
        )
    else:
        tooltips.append(("Degree", "@degree"))

    if show_role:
        tooltips.append(("Type", "@role"))
    if show_value:
        tooltips.append(("Value", "@value{0.0000}"))
    return tooltips


def _network_on_map(
    graph: nx.Graph,
    title: str,
    max_edges: int = 1500,
    node_size: int = 10,
    edge_alpha: float = 0.35,
    node_value: Dict[int, float] | None = None,
    roles: Dict[int, str] | None = None,
    colorbar_title: str = "Value",
    node_palette=Turbo256,
    edge_color: str = "gray",
    directed_arrows: bool = False,
):
    sampled = _subsample_graph_edges(graph, max_edges=max_edges)
    _prepare_node_attributes(sampled, graph, values=node_value, roles=roles)

    pos = _positions(sampled)
    plot = _base_figure(title, pos, tooltips=_node_tooltips(graph, show_value=node_value is not None, show_role=roles is not None))

    network_graph = from_networkx(sampled, pos)
    network_graph.node_renderer.data_source.data["size_base"] = [node_size for _ in sampled.nodes()]
    network_graph.node_renderer.data_source.data["size"] = [node_size for _ in sampled.nodes()]

    if node_value is None:
        network_graph.node_renderer.glyph = Scatter(
            size="size",
            marker="circle",
            fill_color="#1f77b4",
            line_color="white",
            fill_alpha=0.9,
        )
    else:
        vals = [float(v) for v in node_value.values()]
        low, high = min(vals), max(vals)
        if np.isclose(low, high):
            high = low + 1.0
        mapper = LinearColorMapper(palette=node_palette, low=low, high=high)
        network_graph.node_renderer.glyph = Scatter(
            size="size",
            marker="circle",
            fill_color={"field": "value", "transform": mapper},
            line_color="white",
            fill_alpha=0.9,
        )
        color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(), label_standoff=8, title=colorbar_title)
        plot.add_layout(color_bar, "right")

    network_graph.node_renderer.hover_glyph = Scatter(size="size", marker="circle", fill_color="white", line_color="black", line_width=1.8)
    network_graph.node_renderer.selection_glyph = Scatter(size="size", marker="circle", fill_color="white", line_color="black", line_width=1.8)

    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.12, line_width=1.0, line_color="gray")
    network_graph.edge_renderer.hover_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")
    network_graph.edge_renderer.selection_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")

    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(network_graph)
    _attach_zoom_scaling(plot, network_graph.node_renderer.data_source, base_size=node_size)

    if directed_arrows and sampled.is_directed():
        for u, v in sampled.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            arrow = Arrow(
                end=OpenHead(size=6, line_color=edge_color),
                x_start=x0,
                y_start=y0,
                x_end=x1,
                y_end=y1,
                line_color=edge_color,
                line_alpha=0.0,
            )
            plot.add_layout(arrow)

    return plot


def plot_all_connections(graph: nx.DiGraph, max_edges: int = 300):
    return _network_on_map(
        graph,
        title=f"All Airports and Connections (Nodes={graph.number_of_nodes()}, Edges={graph.number_of_edges()})",
        max_edges=max_edges,
        node_size=14,
        directed_arrows=True,
    )


def plot_one_way_connections(graph_one_way: nx.DiGraph, max_edges: int = 500):
    return _network_on_map(
        graph_one_way,
        title="One-Way Airport Connections",
        max_edges=max_edges,
        node_size=14,
        directed_arrows=True,
    )


def plot_one_way_degree_diff(graph_one_way: nx.DiGraph, max_edges: int = 500):
    degree_diffs = compute_degree_differences(graph_one_way)
    return _network_on_map(
        graph_one_way,
        title="One-Way Connections Colored by (In Degree - Out Degree)",
        max_edges=max_edges,
        node_value=degree_diffs,
        colorbar_title="In - Out",
        node_size=14,
        directed_arrows=True,
    )


def plot_two_way_connections(graph_undirected: nx.Graph, max_edges: int = 500):
    return _network_on_map(
        graph_undirected,
        title=f"Bidirectional Airport Connections (Nodes={graph_undirected.number_of_nodes()}, Edges={graph_undirected.number_of_edges()})",
        max_edges=max_edges,
        node_size=14,
        directed_arrows=False,
    )


def plot_centrality_grid(graph_undirected: nx.Graph, max_edges_each: int = 600):
    centralities = compute_centralities(graph_undirected)
    figures = []
    for title, values in centralities.items():
        fig = _network_on_map(
            graph_undirected,
            title=title,
            max_edges=max_edges_each,
            node_value=values,
            colorbar_title=title,
            node_size=13,
            directed_arrows=False,
        )
        figures.append(fig)

    return column(*figures)


def plot_degree_histogram(graph_undirected: nx.Graph):
    degrees = [d for _, d in graph_undirected.degree()]
    hist, edges = np.histogram(degrees, bins=40)

    fig = figure(
        title="Node Degree Histogram (Undirected Graph)",
        width=950,
        height=450,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#1f77b4", line_color="white", alpha=0.85)
    fig.xaxis.axis_label = "Degree"
    fig.yaxis.axis_label = "Frequency"
    return fig


def plot_local_clustering_map(graph_undirected: nx.Graph, max_edges: int = 1200):
    coeff = nx.clustering(graph_undirected)
    return _network_on_map(
        graph_undirected,
        title="Local Clustering Coefficient Map",
        max_edges=max_edges,
        node_value=coeff,
        colorbar_title="Clustering",
        node_size=13,
        directed_arrows=False,
    )


def plot_center_periphery(graph_undirected: nx.Graph, max_edges: int = 1200):
    components = list(nx.connected_components(graph_undirected))
    if not components:
        raise ValueError("The undirected graph has no connected components.")

    largest_cc = max(components, key=len)
    g_lcc = graph_undirected.subgraph(largest_cc).copy()

    center_nodes = set(nx.center(g_lcc))
    periphery_nodes = set(nx.periphery(g_lcc))

    values = {}
    roles = {}
    for n in g_lcc.nodes():
        if n in center_nodes:
            values[n] = 2.0
            roles[n] = "center"
        elif n in periphery_nodes:
            values[n] = 1.0
            roles[n] = "periphery"
        else:
            values[n] = 0.0
            roles[n] = "other"

    fig = _network_on_map(
        g_lcc,
        title="Center (2), Periphery (1), Other (0) - Largest Connected Component",
        max_edges=max_edges,
        node_value=values,
        roles=roles,
        colorbar_title="Role",
        node_size=13,
        directed_arrows=False,
    )
    return fig


def plot_degree_distribution_loglog(graph_undirected: nx.Graph, add_fit: bool = False):
    degrees = np.array([d for _, d in graph_undirected.degree()])
    if degrees.size == 0:
        raise ValueError("No degree values available for log-log plot.")

    unique_degrees, counts = np.unique(degrees, return_counts=True)
    pmf = counts / counts.sum()

    mask = (unique_degrees > 0) & (pmf > 0)
    x = unique_degrees[mask]
    y = pmf[mask]

    fig = figure(
        title="Degree Distribution PMF (log-log)",
        width=950,
        height=500,
        x_axis_type="log",
        y_axis_type="log",
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    source = ColumnDataSource(dict(x=x, y=y))
    fig.circle(x="x", y="y", size=7, alpha=0.8, source=source, color="black")

    if add_fit and len(x) >= 2:
        slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 150)
        y_fit = (10 ** intercept) * (x_fit ** slope)
        fig.line(x_fit, y_fit, line_dash="dashed", line_color="red", line_width=2)

    fig.xaxis.axis_label = "k"
    fig.yaxis.axis_label = "P(k)"
    return fig


def run_pipeline(config: AirportDatasetConfig):
    """High-level helper that returns all core objects used in the notebook."""
    nodes_df, edges_df = load_airport_data(config)
    graph_directed = build_directed_graph(nodes_df, edges_df, config)
    graph_one_way, graph_undirected = split_one_way_and_two_way(graph_directed)

    directed_table, undirected_table = build_summary_tables(graph_directed, graph_undirected)

    return {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "G": graph_directed,
        "G_one_way": graph_one_way,
        "G_undirected": graph_undirected,
        "directed_table": directed_table,
        "undirected_table": undirected_table,
    }
