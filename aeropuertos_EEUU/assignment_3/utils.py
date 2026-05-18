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
from bokeh.palettes import Category20, Category10, Turbo256, Viridis256
from bokeh.plotting import figure, from_networkx
from bokeh.transform import factor_cmap

try:
    import igraph as ig
except Exception:  # pragma: no cover - optional dependency
    ig = None

try:
    import leidenalg
except Exception:  # pragma: no cover - optional dependency
    leidenalg = None

from networkx.algorithms.community import girvan_newman, louvain_communities, modularity


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


_CENTER_PERIPHERY_FACTORS = ["center", "periphery", "other"]
_CENTER_PERIPHERY_PALETTE = ["#d62728", "#1f77b4", "#9e9e9e"]
_CENTER_PERIPHERY_LABELS = ["Center", "Periphery", "Other"]


def _attach_zoom_scaling(plot, source, base_size: int, min_size: int = 6, max_size: int = 20):
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


def _attach_incident_edge_highlighting(plot, network_graph, pos: Dict[int, Tuple[float, float]]):
    """Highlight edges incident to the node under the cursor (network map plots only)."""
    node_source = network_graph.node_renderer.data_source
    edge_source = network_graph.edge_renderer.data_source

    node_ids = node_source.data.get("index", list(pos.keys()))
    node_source.data["x"] = [pos[n][0] for n in node_ids]
    node_source.data["y"] = [pos[n][1] for n in node_ids]

    n_edges = len(edge_source.data.get("start", []))
    edge_source.data["line_alpha"] = [0.12] * n_edges
    edge_source.data["line_color"] = ["gray"] * n_edges

    network_graph.edge_renderer.glyph = MultiLine(line_alpha="line_alpha", line_width=1.0, line_color="line_color")
    network_graph.edge_renderer.hover_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")
    network_graph.edge_renderer.selection_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")

    mouse_callback = CustomJS(
        args=dict(node_source=node_source, edge_source=edge_source, x_range=plot.x_range, y_range=plot.y_range),
        code="""
        const ns = node_source.data;
        const es = edge_source.data;
        const mx = cb_obj.x;
        const my = cb_obj.y;
        const N = (ns['x'] || []).length;
        if (N === 0) { return; }

        const x_span = x_range.end - x_range.start;
        const threshold = Math.pow(x_span / 80.0, 2);

        let minD = Infinity;
        let minIdx = -1;
        for (let i = 0; i < N; i++) {
            const dx = ns['x'][i] - mx;
            const dy = ns['y'][i] - my;
            const d = dx * dx + dy * dy;
            if (d < minD) { minD = d; minIdx = i; }
        }

        function reset_edges() {
            for (let j = 0; j < (es['start'] || []).length; j++) {
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        if (minD > threshold) {
            reset_edges();
            edge_source.change.emit();
            return;
        }

        const node_id = ns['index'][minIdx];
        reset_edges();
        for (let j = 0; j < (es['start'] || []).length; j++) {
            if (es['start'][j] === node_id || es['end'][j] === node_id) {
                es['line_alpha'][j] = 0.95;
                es['line_color'][j] = '#111111';
            }
        }
        edge_source.change.emit();
        """,
    )

    leave_callback = CustomJS(
        args=dict(edge_source=edge_source),
        code="""
        const es = edge_source.data;
        for (let j = 0; j < (es['start'] || []).length; j++) {
            es['line_alpha'][j] = 0.12;
            es['line_color'][j] = 'gray';
        }
        edge_source.change.emit();
        """,
    )

    plot.js_on_event("mousemove", mouse_callback)
    plot.js_on_event("mouseleave", leave_callback)


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
    node_size: int = 8,
    edge_alpha: float = 0.35,
    node_value: Dict[int, float] | None = None,
    roles: Dict[int, str] | None = None,
    colorbar_title: str = "Value",
    node_palette=Turbo256,
    categorical_coloring: bool = False,
    edge_color: str = "gray",
    directed_arrows: bool = False,
):
    sampled = _subsample_graph_edges(graph, max_edges=max_edges)
    _prepare_node_attributes(sampled, graph, values=node_value, roles=roles)

    pos = _positions(sampled)
    plot = _base_figure(
        title,
        pos,
        tooltips=_node_tooltips(
            graph,
            show_value=node_value is not None and not categorical_coloring,
            show_role=roles is not None,
        ),
    )

    network_graph = from_networkx(sampled, pos)
    network_graph.node_renderer.data_source.data["size_base"] = [node_size for _ in sampled.nodes()]
    network_graph.node_renderer.data_source.data["size"] = [node_size for _ in sampled.nodes()]

    if categorical_coloring and roles is not None:
        network_graph.node_renderer.glyph = Scatter(
            size="size",
            marker="circle",
            fill_color=factor_cmap("role", palette=_CENTER_PERIPHERY_PALETTE, factors=_CENTER_PERIPHERY_FACTORS),
            line_color="white",
            fill_alpha=0.9,
        )
        for label, color in zip(_CENTER_PERIPHERY_LABELS, _CENTER_PERIPHERY_PALETTE):
            plot.scatter([], [], legend_label=label, fill_color=color, line_color="white", size=node_size)
        plot.legend.location = "top_right"
        plot.legend.click_policy = "hide"
        plot.legend.background_fill_alpha = 0.85
    elif node_value is None:
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

    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.12, line_width=1.0, line_color="gray")
    network_graph.edge_renderer.hover_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")
    network_graph.edge_renderer.selection_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")

    plot.renderers.append(network_graph)
    _attach_incident_edge_highlighting(plot, network_graph, pos)
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
        node_size=10,
        directed_arrows=True,
    )


def plot_one_way_connections(graph_one_way: nx.DiGraph, max_edges: int = 500):
    return _network_on_map(
        graph_one_way,
        title="One-Way Airport Connections",
        max_edges=max_edges,
        node_size=10,
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
        node_size=10,
        directed_arrows=True,
    )


def plot_two_way_connections(graph_undirected: nx.Graph, max_edges: int = 500):
    return _network_on_map(
        graph_undirected,
        title=f"Bidirectional Airport Connections (Nodes={graph_undirected.number_of_nodes()}, Edges={graph_undirected.number_of_edges()})",
        max_edges=max_edges,
        node_size=10,
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
            node_palette=Viridis256,
            node_size=9,
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
        node_palette=Viridis256,
        node_size=9,
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

    roles = {}
    for n in g_lcc.nodes():
        if n in center_nodes:
            roles[n] = "center"
        elif n in periphery_nodes:
            roles[n] = "periphery"
        else:
            roles[n] = "other"

    fig = _network_on_map(
        g_lcc,
        title="Center, Periphery, and Other — Largest Connected Component",
        max_edges=max_edges,
        roles=roles,
        categorical_coloring=True,
        node_size=9,
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


def _communities_to_mapping(communities):
    community_membership = {}
    for index, community in enumerate(communities, start=1):
        for node in community:
            community_membership[node] = f"Community {index}"
    return community_membership


def _community_palette(count: int):
    if count <= 0:
        return []
    if count == 1:
        return [Category10[3][0]]
    if count == 2:
        # use a red/green pair for clear contrast
        return ["#e41a1c", "#4daf4a"]
    if count <= 3:
        return Category10[3][:count]
    if count <= 10:
        # Category10 provides a distinct qualitative palette for up to 10 categories
        return Category10[10][:count]
    if count <= 20:
        return Category20[20][:count]
    return [Turbo256[int(i)] for i in np.linspace(0, len(Turbo256) - 1, count).astype(int)]


def _plot_community_map(
    graph: nx.Graph,
    communities,
    title: str,
    modularity_score: float,
    max_edges: int = 1200,
    node_size: int = 9,
):
    community_membership = _communities_to_mapping(communities)
    sampled = _subsample_graph_edges(graph, max_edges=max_edges)
    _prepare_node_attributes(sampled, graph, roles={n: community_membership.get(n, "Community 0") for n in sampled.nodes()})

    pos = _positions(sampled)
    plot = _base_figure(
        title,
        pos,
        tooltips=[("Airport", "@name"), ("Community", "@role"), ("Degree", "@degree")],
    )

    network_graph = from_networkx(sampled, pos)
    labels = [community_membership.get(node, "Community 0") for node in sampled.nodes()]
    factors = sorted(set(labels))
    palette = _community_palette(len(factors))

    # assign a color per community (categorical)
    color_map = {f: palette[i % len(palette)] for i, f in enumerate(factors)}
    colors = [color_map[l] for l in labels]

    node_source = network_graph.node_renderer.data_source
    edge_source = network_graph.edge_renderer.data_source

    node_source.data["community"] = labels
    # expose 'role' for existing tooltips compatibility
    node_source.data["role"] = labels
    # ensure index mapping exists (node ids) for JS
    node_source.data["index"] = list(sampled.nodes())
    # ensure x/y present for direct access from JS
    node_source.data["x"] = [pos[n][0] for n in sampled.nodes()]
    node_source.data["y"] = [pos[n][1] for n in sampled.nodes()]
    node_source.data["color"] = colors
    node_source.data["orig_color"] = list(colors)
    node_source.data["size_base"] = [node_size for _ in sampled.nodes()]
    node_source.data["size"] = [node_size for _ in sampled.nodes()]
    node_source.data["alpha"] = [0.9 for _ in sampled.nodes()]

    # use per-node color and alpha so we can update them from JS on hover
    network_graph.node_renderer.glyph = Scatter(
        size="size",
        marker="circle",
        fill_color="color",
        line_color="white",
        fill_alpha="alpha",
    )
    network_graph.node_renderer.hover_glyph = Scatter(size="size", marker="circle", fill_color="white", line_color="black", line_width=1.8)
    network_graph.node_renderer.selection_glyph = Scatter(size="size", marker="circle", fill_color="white", line_color="black", line_width=1.8)

    # prepare edge columns for JS-driven highlighting
    n_edges = len(edge_source.data.get("start", []))
    edge_source.data.setdefault("line_alpha", [0.12] * n_edges)
    edge_source.data.setdefault("line_color", ["gray"] * n_edges)

    network_graph.edge_renderer.glyph = MultiLine(line_alpha="line_alpha", line_width=1.0, line_color="line_color")
    network_graph.edge_renderer.hover_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")
    network_graph.edge_renderer.selection_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")

    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

    # JS callback: when hovering a node, highlight all nodes and intra-community edges
    callback = CustomJS(
        args=dict(node_source=node_source, edge_source=edge_source),
        code="""
        const ns = node_source.data;
        const es = edge_source.data;
        // get hovered indices from inspected (hover) or selected (fallback)
        const inspected = (node_source.inspected && node_source.inspected.indices) ? node_source.inspected.indices : (node_source.selected ? node_source.selected.indices : []);
        const N = (ns['community'] || []).length;
        const ids = ns['index'] || [];

        // build id -> community and id -> orig_color maps
        const id2comm = {};
        const id2orig = {};
        for (let i = 0; i < N; i++) {
            const id = ids[i];
            id2comm[id] = ns['community'][i];
            id2orig[id] = ns['orig_color'][i];
        }

        // reset helpers
        function reset_all(){
            for (let i = 0; i < N; i++){
                ns['alpha'][i] = 0.9;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
            for (let j = 0; j < (es['start'] || []).length; j++){
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        if (!inspected || inspected.length === 0) {
            reset_all();
        } else {
            reset_all();
            const pick = inspected[0];
            const target_comm = ns['community'][pick];
            // highlight nodes in community
            for (let i = 0; i < N; i++){
                if (ns['community'][i] === target_comm){
                    ns['alpha'][i] = 0.95;
                    ns['size'][i] = ns['size_base'][i] * 1.6;
                    ns['color'][i] = ns['orig_color'][i];
                } else {
                    ns['alpha'][i] = 0.15;
                    ns['size'][i] = ns['size_base'][i];
                    // keep original color but dim via alpha
                    ns['color'][i] = ns['orig_color'][i];
                }
            }
            // highlight intra-community edges
            for (let j = 0; j < (es['start'] || []).length; j++){
                const s = es['start'][j];
                const t = es['end'][j];
                const s_comm = id2comm[s];
                const t_comm = id2comm[t];
                if (s_comm === target_comm && t_comm === target_comm){
                    es['line_alpha'][j] = 0.9;
                    es['line_color'][j] = id2orig[s] || 'black';
                } else {
                    es['line_alpha'][j] = 0.12;
                    es['line_color'][j] = 'gray';
                }
            }
        }

        node_source.change.emit();
        edge_source.change.emit();
        """,
    )


    hover = HoverTool(tooltips=[("Airport", "@name"), ("Community", "@community"), ("Degree", "@degree")], renderers=[network_graph.node_renderer], mode='mouse')
    plot.add_tools(hover)

    # Robust interaction: compute nearest node from mouse position and highlight its community.
    mouse_callback = CustomJS(
        args=dict(node_source=node_source, edge_source=edge_source, x_range=plot.x_range, y_range=plot.y_range),
        code="""
        const ns = node_source.data;
        const es = edge_source.data;
        const mx = cb_obj.x;
        const my = cb_obj.y;
        const N = (ns['x'] || []).length;
        if (N === 0) { return; }

        // squared distance threshold based on current view
        const x_span = x_range.end - x_range.start;
        const threshold = Math.pow(x_span / 80.0, 2);

        // find nearest node
        let minD = Infinity;
        let minIdx = -1;
        for (let i = 0; i < N; i++) {
            const dx = ns['x'][i] - mx;
            const dy = ns['y'][i] - my;
            const d = dx * dx + dy * dy;
            if (d < minD) { minD = d; minIdx = i; }
        }

        function reset_all(){
            for (let i = 0; i < N; i++){
                ns['alpha'][i] = 0.9;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
            for (let j = 0; j < (es['start'] || []).length; j++){
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        if (minD > threshold) {
            reset_all();
            node_source.change.emit();
            edge_source.change.emit();
            console.log('community-mousemove: no node within threshold')
            return;
        }

        // highlight community of nearest node
        const target_comm = ns['community'][minIdx];
        for (let i = 0; i < N; i++){
            if (ns['community'][i] === target_comm){
                ns['alpha'][i] = 0.95;
                ns['size'][i] = ns['size_base'][i] * 1.6;
                ns['color'][i] = ns['orig_color'][i];
            } else {
                ns['alpha'][i] = 0.15;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
        }
        for (let j = 0; j < (es['start'] || []).length; j++){
            const s = es['start'][j];
            const t = es['end'][j];
            const s_comm = ns['community'][ ns['index'].indexOf(s) ];
            const t_comm = ns['community'][ ns['index'].indexOf(t) ];
            if (s_comm === target_comm && t_comm === target_comm){
                es['line_alpha'][j] = 0.9;
                es['line_color'][j] = ns['orig_color'][ ns['index'].indexOf(s) ] || 'black';
            } else {
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        console.log('community-mousemove: highlight', minIdx, target_comm)
        node_source.change.emit();
        edge_source.change.emit();
        """,
    )

    leave_callback = CustomJS(
        args=dict(node_source=node_source, edge_source=edge_source),
        code="""
        const ns = node_source.data;
        const es = edge_source.data;
        const N = (ns['community'] || []).length;
        for (let i = 0; i < N; i++){
            ns['alpha'][i] = 0.9;
            ns['size'][i] = ns['size_base'][i];
            ns['color'][i] = ns['orig_color'][i];
        }
        for (let j = 0; j < (es['start'] || []).length; j++){
            es['line_alpha'][j] = 0.12;
            es['line_color'][j] = 'gray';
        }
        console.log('community-mouseleave: reset')
        node_source.change.emit();
        edge_source.change.emit();
        """,
    )

    plot.js_on_event('mousemove', mouse_callback)
    plot.js_on_event('mouseleave', leave_callback)

    # selection (tap) callback: highlight community when a node is selected (click)
    selection_callback = CustomJS(
        args=dict(node_source=node_source, edge_source=edge_source),
        code="""
        const ns = node_source.data;
        const es = edge_source.data;
        const sel = node_source.selected ? node_source.selected.indices : [];
        const N = (ns['community'] || []).length;

        function reset_all(){
            for (let i = 0; i < N; i++){
                ns['alpha'][i] = 0.9;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
            for (let j = 0; j < (es['start'] || []).length; j++){
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        if (!sel || sel.length === 0) {
            reset_all();
        } else {
            // expand selection to full community
            const pick = sel[0];
            const target_comm = ns['community'][pick];
            const matched = [];
            for (let i = 0; i < N; i++){
                if (ns['community'][i] === target_comm) matched.push(i);
            }
            // set the selected indices to the whole community
            if (node_source.selected) {
                node_source.selected.indices = matched;
            }

            reset_all();
            for (let i = 0; i < N; i++){
                if (ns['community'][i] === target_comm){
                    ns['alpha'][i] = 0.95;
                    ns['size'][i] = ns['size_base'][i] * 1.6;
                    ns['color'][i] = ns['orig_color'][i];
                } else {
                    ns['alpha'][i] = 0.15;
                    ns['size'][i] = ns['size_base'][i];
                    ns['color'][i] = ns['orig_color'][i];
                }
            }
            for (let j = 0; j < (es['start'] || []).length; j++){
                const s = es['start'][j];
                const t = es['end'][j];
                const s_idx = ns['index'].indexOf(s);
                const t_idx = ns['index'].indexOf(t);
                const s_comm = s_idx >= 0 ? ns['community'][s_idx] : null;
                const t_comm = t_idx >= 0 ? ns['community'][t_idx] : null;
                if (s_comm === target_comm && t_comm === target_comm){
                    es['line_alpha'][j] = 0.9;
                    es['line_color'][j] = ns['orig_color'][s_idx] || 'black';
                } else {
                    es['line_alpha'][j] = 0.12;
                    es['line_color'][j] = 'gray';
                }
            }
        }

        node_source.change.emit();
        edge_source.change.emit();
        """,
    )

    try:
        node_source.selected.js_on_change('indices', selection_callback)
    except Exception:
        pass

    plot.renderers.append(network_graph)
    _attach_zoom_scaling(plot, network_graph.node_renderer.data_source, base_size=node_size)
    return plot, modularity_score


def louvain_communities_plot(graph_undirected: nx.Graph, max_edges: int = 1200):
    communities = louvain_communities(graph_undirected, seed=42, weight="weight")
    score = modularity(graph_undirected, communities, weight="weight")
    return _plot_community_map(graph_undirected, communities, "Louvain Communities", score, max_edges=max_edges)


def leiden_communities_plot(graph_undirected: nx.Graph, max_edges: int = 1200):
    if ig is not None:
        ig_graph = ig.Graph.from_networkx(graph_undirected)
        vertex_names = ig_graph.vs["name"]
        if hasattr(ig_graph, "community_leiden"):
            partition = ig_graph.community_leiden(objective_function="modularity")
            communities = []
            for community_index in range(len(partition)):
                members = {name for name, membership in zip(vertex_names, partition.membership) if membership == community_index}
                if members:
                    communities.append(members)
            score = partition.modularity
            return _plot_community_map(graph_undirected, communities, "Leiden Communities", score, max_edges=max_edges)

        if leidenalg is not None:
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            communities = []
            for community_index in range(len(partition)):
                members = {name for name, membership in zip(vertex_names, partition.membership) if membership == community_index}
                if members:
                    communities.append(members)
            score = partition.modularity
            return _plot_community_map(graph_undirected, communities, "Leiden Communities", score, max_edges=max_edges)

    print("Warning: Leiden algorithm not available, falling back to Louvain communities.")
    fallback_communities = louvain_communities(graph_undirected, seed=42, weight="weight")
    fallback_score = modularity(graph_undirected, fallback_communities, weight="weight")
    return _plot_community_map(
        graph_undirected,
        fallback_communities,
        "Leiden Communities (fallback partition)",
        fallback_score,
        max_edges=max_edges,
    )


def girvan_newman_communities_plot(graph_undirected: nx.Graph, max_edges: int = 1200, max_levels: int = 6):
    best_partition = None
    best_score = -1.0
    best_communities = None

    for level, partition in enumerate(girvan_newman(graph_undirected), start=1):
        communities = [set(group) for group in partition]
        score = modularity(graph_undirected, communities)
        if score > best_score:
            best_score = score
            best_partition = level
            best_communities = communities
        if level >= max_levels:
            break

    if best_communities is None:
        best_communities = [set(graph_undirected.nodes())]
        best_score = 0.0

    return _plot_community_map(
        graph_undirected,
        best_communities,
        f"Girvan-Newman Communities (best level {best_partition or 1})",
        best_score,
        max_edges=max_edges,
    )


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
