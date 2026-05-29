from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import (
    Arrow,
    BasicTicker,
    Circle,
    ColumnDataSource,
    ColorBar,
    CustomJS,
    HoverTool,
    LabelSet,
    LinearColorMapper,
    MultiLine,
    NodesOnly,
    OpenHead,
    Scatter,
    WMTSTileSource,
)
from bokeh.palettes import Category20, Category10, Turbo256, Viridis256
from bokeh.plotting import figure, from_networkx
from bokeh.transform import factor_cmap

try:
    import igrzaph as ig
except Exception:  # pragma: no cover - optional dependency
    ig = None

try:
    import leidenalg
except Exception:  # pragma: no cover - optional dependency
    leidenalg = None

from networkx.algorithms.community import girvan_newman, louvain_communities, modularity
from networkx.algorithms.community.louvain import louvain_partitions


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


def _add_categorical_legend(plot, labels: List[str], colors: List[str], marker_size: int):
    """Add a legend with visible color markers (empty scatters omit swatches in Bokeh)."""
    x_span = float(plot.x_range.end - plot.x_range.start)
    y_span = float(plot.y_range.end - plot.y_range.start)
    legend_x = float(plot.x_range.start) - x_span
    legend_y = float(plot.y_range.start) - y_span

    for label, color in zip(labels, colors):
        plot.scatter(
            [legend_x],
            [legend_y],
            legend_label=label,
            fill_color=color,
            line_color="white",
            size=marker_size,
            fill_alpha=0.9,
        )

    if plot.legend is not None:
        plot.legend.location = "top_right"
        plot.legend.click_policy = "hide"
        plot.legend.background_fill_alpha = 0.85
        plot.legend.label_text_font_size = "10pt"
        plot.legend.glyph_width = 18
        plot.legend.glyph_height = 18


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
        _add_categorical_legend(plot, _CENTER_PERIPHERY_LABELS, _CENTER_PERIPHERY_PALETTE, node_size)
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


# === world ariports utils ===

def _layout_positions(graph: nx.Graph, seed: int = 42) -> Dict[int, Tuple[float, float]]:
    return nx.spring_layout(graph, seed=seed)


def _base_layout_figure(title: str, pos: Dict[int, Tuple[float, float]]):
    x_vals = [xy[0] for xy in pos.values()]
    y_vals = [xy[1] for xy in pos.values()]
    pad_x = (max(x_vals) - min(x_vals)) * 0.08 if x_vals else 1.0
    pad_y = (max(y_vals) - min(y_vals)) * 0.08 if y_vals else 1.0

    plot = figure(
        title=title,
        tools="pan,wheel_zoom,box_zoom,save,reset,tap",
        active_scroll="wheel_zoom",
        x_range=(min(x_vals) - pad_x, max(x_vals) + pad_x) if x_vals else (-1, 1),
        y_range=(min(y_vals) - pad_y, max(y_vals) + pad_y) if y_vals else (-1, 1),
        width=900,
        height=600,
    )
    plot.axis.visible = False
    plot.grid.visible = False
    return plot


def _network_on_layout(
    graph: nx.Graph,
    title: str,
    max_edges: int = 2000,
    node_size: int = 8,
    node_value: Dict[int, float] | None = None,
    roles: Dict[int, str] | None = None,
    colorbar_title: str = "Value",
    node_palette=Turbo256,
    categorical_coloring: bool = False,
    seed: int = 42,
):
    sampled = _subsample_graph_edges(graph, max_edges=max_edges)
    _prepare_node_attributes(sampled, graph, values=node_value, roles=roles)

    pos = _layout_positions(sampled, seed=seed)
    plot = _base_layout_figure(title, pos)

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
        _add_categorical_legend(plot, _CENTER_PERIPHERY_LABELS, _CENTER_PERIPHERY_PALETTE, node_size)
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
    return plot

def load_world_edge_list(edges_path: str | Path, comment_prefix: str = "%") -> pd.DataFrame:
    """Load world edge list with source/target columns."""
    edges_df = pd.read_csv(
        edges_path,
        comment=comment_prefix,
        sep=r"\s+",
        header=None,
        names=["source", "target"],
    )
    return edges_df.astype({"source": int, "target": int})


def build_world_graphs(edges_df: pd.DataFrame) -> Tuple[nx.DiGraph, nx.Graph]:
    """Build directed and undirected world graphs (undirected if any directed edge exists)."""
    directed = nx.DiGraph()
    directed.add_edges_from((int(s), int(t)) for s, t in edges_df[["source", "target"]].itertuples(index=False, name=None))

    for node in directed.nodes():
        directed.nodes[node]["name"] = str(node)

    undirected = nx.Graph()
    undirected.add_nodes_from(directed.nodes(data=True))
    undirected.add_edges_from({tuple(sorted((u, v))) for u, v in directed.edges() if u != v})

    return directed, undirected


def plot_world_force_layout(graph: nx.Graph, max_edges: int = 2000, seed: int = 42):
    """Plot a force-directed layout for the world graph (edge sampling supported)."""
    edge_list = list(graph.edges())
    if len(edge_list) > max_edges:
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(len(edge_list), size=max_edges, replace=False)
        sampled_edges = [edge_list[i] for i in sampled_idx]
        sampled_graph = nx.Graph()
        sampled_graph.add_nodes_from(graph.nodes(data=True))
        sampled_graph.add_edges_from(sampled_edges)
    else:
        sampled_graph = graph

    pos = nx.spring_layout(sampled_graph, seed=seed)
    plot = figure(title="World Airports - Force-Directed Layout (sampled)", width=900, height=600)
    plot.axis.visible = False
    plot.grid.visible = False

    plot_graph = from_networkx(sampled_graph, pos)
    plot_graph.node_renderer.glyph = Circle(size=6, fill_color="#1f77b4", line_color="white", fill_alpha=0.8)
    plot_graph.edge_renderer.glyph = MultiLine(line_alpha=0.15, line_width=1.0, line_color="#888888")
    plot.renderers.append(plot_graph)
    return plot


def plot_world_directed_overview(graph: nx.DiGraph, max_edges: int = 2000, seed: int = 42):
    return _network_on_layout(
        graph,
        title=f"World Directed Graph (Nodes={graph.number_of_nodes()}, Edges={graph.number_of_edges()})",
        max_edges=max_edges,
        node_size=8,
        seed=seed,
    )


def plot_world_undirected_overview(graph: nx.Graph, max_edges: int = 2000, seed: int = 42):
    return _network_on_layout(
        graph,
        title=f"World Undirected Graph (Nodes={graph.number_of_nodes()}, Edges={graph.number_of_edges()})",
        max_edges=max_edges,
        node_size=8,
        seed=seed,
    )


def plot_world_local_clustering(graph: nx.Graph, max_edges: int = 2000, seed: int = 42):
    coeff = nx.clustering(graph)
    return _network_on_layout(
        graph,
        title="Local Clustering Coefficient (World)",
        max_edges=max_edges,
        node_value=coeff,
        colorbar_title="Clustering",
        node_palette=Viridis256,
        node_size=8,
        seed=seed,
    )


def plot_world_center_periphery(graph: nx.Graph, max_edges: int = 2000, seed: int = 42):
    components = list(nx.connected_components(graph))
    if not components:
        raise ValueError("The world graph has no connected components.")

    largest_cc = max(components, key=len)
    g_lcc = graph.subgraph(largest_cc).copy()

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

    return _network_on_layout(
        g_lcc,
        title="Center, Periphery, and Other — Largest Connected Component",
        max_edges=max_edges,
        roles=roles,
        categorical_coloring=True,
        node_size=8,
        seed=seed,
    )


def plot_world_louvain_communities(graph: nx.Graph, max_edges: int = 2000, seed: int = 42):
    communities = louvain_communities(graph, seed=seed, weight="weight")
    membership = _communities_to_mapping(communities)
    sampled = _subsample_graph_edges(graph, max_edges=max_edges)
    _prepare_node_attributes(sampled, graph, roles={n: membership.get(n, "Community 0") for n in sampled.nodes()})

    pos = _layout_positions(sampled, seed=seed)
    plot = _base_layout_figure("Louvain Communities (World)", pos)
    network_graph = from_networkx(sampled, pos)

    labels = [membership.get(node, "Community 0") for node in sampled.nodes()]
    factors = sorted(set(labels))
    palette = _community_palette(len(factors))
    color_map = {f: palette[i % len(palette)] for i, f in enumerate(factors)}
    colors = [color_map[l] for l in labels]

    node_source = network_graph.node_renderer.data_source
    node_source.data["role"] = labels
    node_source.data["color"] = colors
    node_source.data["size_base"] = [8 for _ in sampled.nodes()]
    node_source.data["size"] = [8 for _ in sampled.nodes()]

    network_graph.node_renderer.glyph = Scatter(
        size="size",
        marker="circle",
        fill_color="color",
        line_color="white",
        fill_alpha=0.9,
    )
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.12, line_width=1.0, line_color="gray")
    network_graph.edge_renderer.hover_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")
    network_graph.edge_renderer.selection_glyph = MultiLine(line_alpha=0.95, line_width=4.0, line_color="#111111")

    plot.renderers.append(network_graph)
    _attach_incident_edge_highlighting(plot, network_graph, pos)
    _attach_zoom_scaling(plot, network_graph.node_renderer.data_source, base_size=8)
    legend_factors = sorted(
        factors,
        key=lambda label: int(label.split()[-1]) if label.startswith("Community ") and label.split()[-1].isdigit() else label,
    )
    legend_colors = [color_map[label] for label in legend_factors]
    _add_categorical_legend(plot, legend_factors, legend_colors, 8)
    return plot


def compute_world_centralities(graph: nx.Graph) -> pd.DataFrame:
    """Compute centrality measures for the world graph."""
    centralities = compute_centralities(graph)
    return pd.DataFrame(centralities)


def top_centrality_nodes(centrality_df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.Series]:
    """Return the top-N nodes per centrality column."""
    return {col: centrality_df[col].sort_values(ascending=False).head(top_n) for col in centrality_df.columns}


def plot_world_degree_histogram(graph: nx.Graph, bins: int = 30):
    """Plot the degree histogram for the world graph."""
    degrees = np.array([d for _, d in graph.degree()])
    hist, edges = np.histogram(degrees, bins=bins)

    fig = figure(
        title="World Graph Degree Histogram",
        width=750,
        height=450,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#1f77b4", line_color="white", alpha=0.85)
    fig.xaxis.axis_label = "Degree"
    fig.yaxis.axis_label = "Count"
    return fig


def plot_world_clustering_histogram(graph: nx.Graph, bins: int = 30) -> Tuple[figure, float]:
    """Plot clustering coefficient distribution and return the average value."""
    clustering = np.array(list(nx.clustering(graph).values()))
    hist, edges = np.histogram(clustering, bins=bins, range=(0.0, 1.0))

    fig = figure(
        title="Local Clustering Coefficient Distribution",
        width=750,
        height=450,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#2ca02c", line_color="white", alpha=0.85)
    fig.xaxis.axis_label = "Clustering Coefficient"
    fig.yaxis.axis_label = "Count"
    return fig, float(clustering.mean()) if clustering.size else 0.0


def compute_world_center_periphery(graph: nx.Graph) -> Dict[str, object]:
    """Compute center/periphery stats on the largest connected component."""
    components = list(nx.connected_components(graph))
    if not components:
        raise ValueError("The world graph has no connected components.")

    largest_cc = max(components, key=len)
    g_lcc = graph.subgraph(largest_cc).copy()
    eccentricity = nx.eccentricity(g_lcc)
    radius = min(eccentricity.values())
    diameter = max(eccentricity.values())
    center_nodes = [n for n, e in eccentricity.items() if e == radius]
    periphery_nodes = [n for n, e in eccentricity.items() if e == diameter]

    return {
        "lcc_nodes": g_lcc.number_of_nodes(),
        "lcc_edges": g_lcc.number_of_edges(),
        "radius": radius,
        "diameter": diameter,
        "center_nodes": center_nodes,
        "periphery_nodes": periphery_nodes,
    }


def plot_world_degree_distribution_loglog(graph: nx.Graph):
    """Plot degree distribution in log-log scale for the world graph."""
    return plot_degree_distribution_loglog(graph, add_fit=False)


def louvain_world_coassignment(
    graph: nx.Graph,
    n_runs: int = 40,
    seed: int = 42,
    max_nodes: int = 40,
    heatmap_sample_seed: int = 2024,
) -> Tuple[List[set], float, object, List[List[str]]]:
    """Run Louvain and return communities, modularity, and co-assignment heatmap layout."""
    communities, modularity_score, membership = louvain_reference_partition(graph, seed=seed)
    co_layout, _, co_label_groups, _ = plot_louvain_coassignment_heatmaps_pair(
        graph,
        communities=communities,
        membership=membership,
        n_runs=n_runs,
        seed=seed,
        max_nodes=max_nodes,
        heatmap_sample_seed=heatmap_sample_seed,
    )
    return communities, modularity_score, co_layout, co_label_groups

#### END WORLD AIRPORTS UTILS

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

    network_graph.selection_policy = NodesOnly()
    network_graph.inspection_policy = NodesOnly()

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

        function reset_edges() {
            for (let j = 0; j < (es['start'] || []).length; j++) {
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        function reset_nodes() {
            for (let i = 0; i < N; i++) {
                ns['alpha'][i] = 0.9;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
        }

        function highlight_community_nodes(target_comm) {
            for (let i = 0; i < N; i++) {
                if (ns['community'][i] === target_comm) {
                    ns['alpha'][i] = 0.95;
                    ns['size'][i] = ns['size_base'][i] * 1.6;
                    ns['color'][i] = ns['orig_color'][i];
                } else {
                    ns['alpha'][i] = 0.15;
                    ns['size'][i] = ns['size_base'][i];
                    ns['color'][i] = ns['orig_color'][i];
                }
            }
        }

        function apply_pinned_selection() {
            const sel = node_source.selected ? node_source.selected.indices : [];
            if (!sel || sel.length === 0) { return false; }
            highlight_community_nodes(ns['community'][sel[0]]);
            reset_edges();
            return true;
        }

        if (minD > threshold) {
            if (apply_pinned_selection()) {
                node_source.change.emit();
                edge_source.change.emit();
                return;
            }
            reset_nodes();
            reset_edges();
            node_source.change.emit();
            edge_source.change.emit();
            return;
        }

        // highlight community of nearest node (nodes + edges on hover only)
        const target_comm = ns['community'][minIdx];
        highlight_community_nodes(target_comm);
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

        function reset_edges() {
            for (let j = 0; j < (es['start'] || []).length; j++) {
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        reset_edges();
        const sel = node_source.selected ? node_source.selected.indices : [];
        if (sel && sel.length > 0) {
            const target_comm = ns['community'][sel[0]];
            for (let i = 0; i < N; i++) {
                if (ns['community'][i] === target_comm) {
                    ns['alpha'][i] = 0.95;
                    ns['size'][i] = ns['size_base'][i] * 1.6;
                    ns['color'][i] = ns['orig_color'][i];
                } else {
                    ns['alpha'][i] = 0.15;
                    ns['size'][i] = ns['size_base'][i];
                    ns['color'][i] = ns['orig_color'][i];
                }
            }
        } else {
            for (let i = 0; i < N; i++) {
                ns['alpha'][i] = 0.9;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
        }
        node_source.change.emit();
        edge_source.change.emit();
        """,
    )

    plot.js_on_event('mousemove', mouse_callback)
    plot.js_on_event('mouseleave', leave_callback)

    # selection (tap) callback: pin community nodes only; edges stay for hover
    selection_callback = CustomJS(
        args=dict(node_source=node_source, edge_source=edge_source),
        code="""
        const ns = node_source.data;
        const es = edge_source.data;
        const sel = node_source.selected ? node_source.selected.indices : [];
        const N = (ns['community'] || []).length;

        function reset_edges() {
            for (let j = 0; j < (es['start'] || []).length; j++) {
                es['line_alpha'][j] = 0.12;
                es['line_color'][j] = 'gray';
            }
        }

        if (!sel || sel.length === 0) {
            for (let i = 0; i < N; i++) {
                ns['alpha'][i] = 0.9;
                ns['size'][i] = ns['size_base'][i];
                ns['color'][i] = ns['orig_color'][i];
            }
            reset_edges();
        } else {
            const pick = sel[0];
            const target_comm = ns['community'][pick];
            const matched = [];
            for (let i = 0; i < N; i++) {
                if (ns['community'][i] === target_comm) matched.push(i);
            }
            if (node_source.selected) {
                node_source.selected.indices = matched;
            }
            for (let i = 0; i < N; i++) {
                if (ns['community'][i] === target_comm) {
                    ns['alpha'][i] = 0.95;
                    ns['size'][i] = ns['size_base'][i] * 1.6;
                    ns['color'][i] = ns['orig_color'][i];
                } else {
                    ns['alpha'][i] = 0.15;
                    ns['size'][i] = ns['size_base'][i];
                    ns['color'][i] = ns['orig_color'][i];
                }
            }
            reset_edges();
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
    legend_factors = sorted(
        factors,
        key=lambda label: int(label.split()[-1]) if label.startswith("Community ") and label.split()[-1].isdigit() else label,
    )
    legend_colors = [color_map[label] for label in legend_factors]
    _add_categorical_legend(plot, legend_factors, legend_colors, node_size)
    _attach_zoom_scaling(plot, network_graph.node_renderer.data_source, base_size=node_size)
    return plot, modularity_score


def louvain_communities_plot(graph_undirected: nx.Graph, max_edges: int = 1200):
    communities = louvain_communities(graph_undirected, seed=42, weight="weight")
    score = modularity(graph_undirected, communities, weight="weight")
    return _plot_community_map(graph_undirected, communities, "Louvain Communities", score, max_edges=max_edges)

##### INICIO ANALISIS DE PARTICIONES ###################################################################################

def louvain_reference_partition(
    graph_undirected: nx.Graph,
    seed: int = 42,
    weight: str = "weight",
) -> Tuple[List[set], float, Dict[int, str]]:
    """Louvain partition used as reference for stability plots (map, bars, heatmap)."""
    communities = list(louvain_communities(graph_undirected, seed=seed, weight=weight))
    score = float(modularity(graph_undirected, communities, weight=weight))
    membership = _communities_to_mapping(communities)
    return communities, score, membership


def louvain_communities_map_plot(
    graph_undirected: nx.Graph,
    communities: Iterable[set] | None = None,
    modularity_score: float | None = None,
    seed: int = 42,
    max_edges: int = 1200,
    weight: str = "weight",
):
    """Community map plus partition metadata for downstream stability plots."""
    if communities is None:
        communities, modularity_score, _ = louvain_reference_partition(
            graph_undirected,
            seed=seed,
            weight=weight,
        )
    communities = list(communities)
    if modularity_score is None:
        modularity_score = float(modularity(graph_undirected, communities, weight=weight))
    title = f"Louvain Communities (Q = {modularity_score:.4f})"
    plot, _ = _plot_community_map(graph_undirected, communities, title, modularity_score, max_edges=max_edges)
    membership = _communities_to_mapping(communities)
    return plot, modularity_score, communities, membership


def louvain_hierarchical_partitions(
    graph_undirected: nx.Graph,
    seed: int = 42,
    weight: str = "weight",
) -> List[List[set]]:
    return [list(partition) for partition in louvain_partitions(graph_undirected, seed=seed, weight=weight)]


def louvain_pre_final_partition(
    graph_undirected: nx.Graph,
    seed: int = 42,
    weight: str = "weight",
) -> List[set]:
    """Partition at the hierarchical level immediately before the final Louvain partition."""
    partitions = louvain_hierarchical_partitions(graph_undirected, seed=seed, weight=weight)
    if len(partitions) >= 2:
        return partitions[-2]
    if partitions:
        return partitions[-1]
    return [{node} for node in graph_undirected.nodes()]


def louvain_pre_final_communities_map_plot(
    graph_undirected: nx.Graph,
    seed: int = 42,
    max_edges: int = 1200,
    weight: str = "weight",
):
    pre_final = louvain_pre_final_partition(graph_undirected, seed=seed, weight=weight)
    score = float(modularity(graph_undirected, pre_final, weight=weight))
    title = f"Louvain pre-final level (Q = {score:.4f}, before chosen partition)"
    plot, _ = _plot_community_map(graph_undirected, pre_final, title, score, max_edges=max_edges)
    return plot, score, pre_final


def _partition_to_labels(graph: nx.Graph, communities: Iterable[set]) -> Dict[int, int]:
    labels: Dict[int, int] = {}
    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id
    for node in graph.nodes():
        labels.setdefault(node, -1)
    return labels


def louvain_coassignment_matrix(
    graph_undirected: nx.Graph,
    n_runs: int = 50,
    seed: int = 0,
    weight: str = "weight",
) -> Tuple[List, np.ndarray]:
    """Fraction of Louvain runs in which each pair of nodes shares a community."""
    nodes = list(graph_undirected.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    coassignment = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    rng = np.random.default_rng(seed)
    run_seeds = rng.integers(0, 2**31 - 1, size=n_runs, dtype=np.int64)

    for run_seed in run_seeds:
        communities = louvain_communities(
            graph_undirected,
            seed=int(run_seed),
            weight=weight,
        )
        labels = np.full(n_nodes, -1, dtype=np.int32)
        for community_id, community in enumerate(communities):
            for node in community:
                labels[node_index[node]] = community_id
        same_community = labels[:, None] == labels[None, :]
        coassignment += same_community.astype(np.float64)

    if n_runs > 0:
        coassignment /= float(n_runs)
    return nodes, coassignment


def _community_index_map(communities: Iterable[set]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for index, community in enumerate(communities, start=1):
        for node in community:
            mapping[node] = index
    return mapping


def _submatrix_and_labels(
    graph_undirected: nx.Graph,
    nodes: List,
    coassignment: np.ndarray,
    selected: List,
    membership: Dict[int, str],
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    index = {node: i for i, node in enumerate(nodes)}
    sub_idx = [index[node] for node in selected]
    sub_matrix = coassignment[np.ix_(sub_idx, sub_idx)]
    tick_labels = []
    community_labels = []
    for node in selected:
        name = graph_undirected.nodes[node].get("name", str(node))
        tick_labels.append(str(name)[:18])
        community_labels.append(membership.get(node, "Community ?"))
    return sub_matrix, tick_labels, community_labels, tick_labels


def _select_nodes_random_uniform(
    graph_undirected: nx.Graph,
    nodes: List,
    communities: Iterable[set],
    max_nodes: int,
    sample_seed: int,
) -> List:
    """Uniformly random nodes from the graph (ordered by reference community)."""
    rng = np.random.default_rng(sample_seed)
    if len(nodes) <= max_nodes:
        selected = list(nodes)
    else:
        selected = list(rng.choice(nodes, size=max_nodes, replace=False))
    comm_index = _community_index_map(communities)
    return sorted(selected, key=lambda n: comm_index.get(n, 10**9))


def _select_nodes_balanced_random_per_community(
    graph_undirected: nx.Graph,
    communities: Iterable[set],
    max_nodes: int,
    sample_seed: int,
) -> List:
    """Equal random sample per community (uniform within each community)."""
    communities = [set(community) for community in communities]
    if not communities:
        return []
    n_communities = len(communities)
    per_community = max(1, max_nodes // n_communities)
    per_community = min(per_community, min(len(community) for community in communities))

    rng = np.random.default_rng(sample_seed)
    selected: List = []
    for community in communities:
        members = list(community)
        if len(members) <= per_community:
            picked = members
        else:
            picked = list(rng.choice(members, size=per_community, replace=False))
        selected.extend(picked)

    comm_index = _community_index_map(communities)
    return sorted(selected, key=lambda n: comm_index.get(n, 10**9))


def _axis_labels_with_community_boundaries(
    tick_labels: List[str],
    community_labels: List[str],
) -> List[str]:
    """Mark community block start only on axis tick labels."""
    if not tick_labels:
        return []
    display = [str(label) for label in tick_labels]
    n = len(display)
    i = 0
    while i < n:
        community = community_labels[i]
        block_start = i
        while i < n and community_labels[i] == community:
            i += 1
        display[block_start] = f"[{community}] {display[block_start]}"
    return display


def _build_coassignment_heatmap_figure(
    title: str,
    matrix: np.ndarray,
    tick_labels: List[str],
    community_labels: List[str],
    n_runs: int,
    mark_community_boundaries_in_labels: bool = False,
    highlight_rows_cols: bool = True,
    figure_size: int = 720,
):
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("No nodes available for the co-assignment heatmap.")

    mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
    fig = figure(
        title=title,
        width=figure_size,
        height=figure_size,
        tools="pan,wheel_zoom,box_zoom,save,reset",
        x_range=(-0.75, n - 0.5),
        y_range=(-0.75, n - 0.5),
        toolbar_location="right",
    )
    axis_labels = (
        _axis_labels_with_community_boundaries(tick_labels, community_labels)
        if mark_community_boundaries_in_labels
        else tick_labels
    )
    fig.xaxis.ticker = list(range(n))
    fig.yaxis.ticker = list(range(n))
    fig.xaxis.major_label_overrides = {i: axis_labels[i] for i in range(n)}
    fig.yaxis.major_label_overrides = {i: axis_labels[i] for i in range(n)}
    fig.xaxis.major_label_orientation = 1.2
    axis_caption = (
        "Airport (community start/end in labels)"
        if mark_community_boundaries_in_labels
        else "Airport index (ordered by community)"
    )
    fig.xaxis.axis_label = axis_caption
    fig.yaxis.axis_label = axis_caption

    cell_source = ColumnDataSource(
        dict(
            x=np.repeat(np.arange(n), n),
            y=np.tile(np.arange(n), n),
            value=matrix.flatten(),
        )
    )
    fig.rect(
        x="x",
        y="y",
        width=1,
        height=1,
        source=cell_source,
        fill_color={"field": "value", "transform": mapper},
        line_color=None,
    )

    airport_source = ColumnDataSource(
        dict(
            x=list(range(n)),
            y=list(range(n)),
            name=tick_labels,
            community=community_labels,
            row_index=list(range(n)),
            col_index=list(range(n)),
        )
    )
    airport_tooltip = [
        ("Airport", "@name"),
        ("Community", "@community"),
        ("Row", "@row_index"),
        ("Column", "@col_index"),
    ]
    diagonal_renderer = fig.scatter(
        x="x",
        y="y",
        size=22,
        source=airport_source,
        fill_alpha=0.01,
        line_alpha=0,
    )
    x_axis_source = ColumnDataSource(
        dict(
            x=list(range(n)),
            y=[-0.4] * n,
            name=tick_labels,
            community=community_labels,
            row_index=list(range(n)),
            col_index=list(range(n)),
        )
    )
    y_axis_source = ColumnDataSource(
        dict(
            x=[-0.4] * n,
            y=list(range(n)),
            name=tick_labels,
            community=community_labels,
            row_index=list(range(n)),
            col_index=list(range(n)),
        )
    )
    x_renderer = fig.scatter(x="x", y="y", size=18, source=x_axis_source, fill_alpha=0.01, line_alpha=0)
    y_renderer = fig.scatter(x="x", y="y", size=18, source=y_axis_source, fill_alpha=0.01, line_alpha=0)
    fig.add_tools(
        HoverTool(
            renderers=[diagonal_renderer, x_renderer, y_renderer],
            tooltips=airport_tooltip,
        )
    )

    if highlight_rows_cols:
        highlight_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
        highlight_renderer = fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=highlight_source,
            fill_color="#ffcc00",
            line_color=None,
            fill_alpha=0.22,
            level="overlay",
        )
        highlight_callback = CustomJS(
            args=dict(
                highlight_source=highlight_source,
                n=n,
                x_range=fig.x_range,
                y_range=fig.y_range,
            ),
            code="""
            const mx = cb_obj.x;
            const my = cb_obj.y;
            const hs = highlight_source.data;
            const x_span = x_range.end - x_range.start;
            const y_span = y_range.end - y_range.start;
            const threshold = Math.max(x_span, y_span) / Math.max(n * 3.0, 1.0);

            let row = Math.round(my);
            let col = Math.round(mx);
            const near_matrix = (
                mx >= -0.5 && mx <= n - 0.5 &&
                my >= -0.5 && my <= n - 0.5
            );
            const near_x_axis = my < -0.1 && mx >= -0.5 && mx <= n - 0.5;
            const near_y_axis = mx < -0.1 && my >= -0.5 && my <= n - 0.5;

            if (!near_matrix && !near_x_axis && !near_y_axis) {
                hs['x'] = [];
                hs['y'] = [];
                hs['width'] = [];
                hs['height'] = [];
                highlight_source.change.emit();
                return;
            }

            if (near_x_axis && !near_matrix) { row = -1; }
            if (near_y_axis && !near_matrix) { col = -1; }

            const rects = {x: [], y: [], width: [], height: []};
            if (row >= 0 && row < n) {
                rects.x.push(-0.75);
                rects.y.push(row - 0.5);
                rects.width.push(n + 0.5);
                rects.height.push(1.0);
            }
            if (col >= 0 && col < n) {
                rects.x.push(col - 0.5);
                rects.y.push(-0.75);
                rects.width.push(1.0);
                rects.height.push(n + 0.5);
            }
            hs['x'] = rects.x;
            hs['y'] = rects.y;
            hs['width'] = rects.width;
            hs['height'] = rects.height;
            highlight_source.change.emit();
            """,
        )
        fig.js_on_event("mousemove", highlight_callback)
        fig.js_on_event("mouseleave", CustomJS(
            args=dict(highlight_source=highlight_source),
            code="""
            const hs = highlight_source.data;
            hs['x'] = [];
            hs['y'] = [];
            hs['width'] = [];
            hs['height'] = [];
            highlight_source.change.emit();
            """,
        ))

    color_bar = ColorBar(color_mapper=mapper, label_standoff=8, title="P(same community)")
    fig.add_layout(color_bar, "right")
    return fig


def plot_louvain_coassignment_heatmap(
    graph_undirected: nx.Graph,
    communities: Iterable[set],
    membership: Dict[int, str],
    n_runs: int = 50,
    seed: int = 0,
    max_nodes: int = 80,
    weight: str = "weight",
    heatmap_sample_seed: int = 2024,
):
    """Single random-sample co-assignment heatmap (see `plot_louvain_coassignment_heatmaps_pair`)."""
    layout, matrices, labels, selected = plot_louvain_coassignment_heatmaps_pair(
        graph_undirected,
        communities=communities,
        membership=membership,
        n_runs=n_runs,
        seed=seed,
        max_nodes=max_nodes,
        weight=weight,
        heatmap_sample_seed=heatmap_sample_seed,
    )
    return layout.children[0], matrices[0], labels[0], selected[0]


def plot_louvain_coassignment_heatmaps_pair(
    graph_undirected: nx.Graph,
    communities: Iterable[set],
    membership: Dict[int, str],
    n_runs: int = 50,
    seed: int = 0,
    max_nodes: int = 80,
    weight: str = "weight",
    heatmap_sample_seed: int = 2024,
):
    """Side-by-side heatmaps: uniform random sample vs balanced random per community."""
    communities = list(communities)

    nodes, coassignment = louvain_coassignment_matrix(
        graph_undirected,
        n_runs=n_runs,
        seed=seed,
        weight=weight,
    )

    random_selected = _select_nodes_random_uniform(
        graph_undirected,
        nodes,
        communities,
        max_nodes=max_nodes,
        sample_seed=heatmap_sample_seed,
    )
    balanced_selected = _select_nodes_balanced_random_per_community(
        graph_undirected,
        communities,
        max_nodes=max_nodes,
        sample_seed=heatmap_sample_seed + 1,
    )

    random_matrix, random_labels, random_communities, _ = _submatrix_and_labels(
        graph_undirected,
        nodes,
        coassignment,
        random_selected,
        membership,
    )
    balanced_matrix, balanced_labels, balanced_communities, _ = _submatrix_and_labels(
        graph_undirected,
        nodes,
        coassignment,
        balanced_selected,
        membership,
    )
    per_comm = len(balanced_selected) // max(1, len(communities))
    fig_random = _build_coassignment_heatmap_figure(
        title=f"Random nodes (uniform sample, max {max_nodes}) — {n_runs} Louvain runs",
        matrix=random_matrix,
        tick_labels=random_labels,
        community_labels=random_communities,
        n_runs=n_runs,
        mark_community_boundaries_in_labels=False,
    )
    fig_balanced = _build_coassignment_heatmap_figure(
        title=(
            f"Balanced random ({per_comm} nodes per community, max {max_nodes}) — "
            f"community start marked in axis labels"
        ),
        matrix=balanced_matrix,
        tick_labels=balanced_labels,
        community_labels=balanced_communities,
        n_runs=n_runs,
        mark_community_boundaries_in_labels=True,
    )

    layout = row(fig_random, fig_balanced)
    return (
        layout,
        [random_matrix, balanced_matrix],
        [random_labels, balanced_labels],
        [random_selected, balanced_selected],
    )


def louvain_community_mean_degrees(
    graph_undirected: nx.Graph,
    communities: Iterable[set] | None = None,
    seed: int = 42,
    weight: str = "weight",
) -> pd.DataFrame:
    """Mean weighted degree per Louvain community (same labels as the map)."""
    if communities is None:
        communities, _, _ = louvain_reference_partition(graph_undirected, seed=seed, weight=weight)
    communities = list(communities)
    rows = []
    palette = _community_palette(len(communities))
    for index, community in enumerate(communities, start=1):
        degrees = [float(graph_undirected.degree(node, weight=weight)) for node in community]
        rows.append(
            {
                "community": f"Community {index}",
                "n_nodes": len(community),
                "mean_degree": float(np.mean(degrees)),
                "color": palette[index - 1],
            }
        )
    return pd.DataFrame(rows)


def plot_louvain_community_mean_degrees(
    graph_undirected: nx.Graph,
    communities: Iterable[set] | None = None,
    seed: int = 42,
    weight: str = "weight",
):
    stats = louvain_community_mean_degrees(
        graph_undirected,
        communities=communities,
        seed=seed,
        weight=weight,
    )
    if stats.empty:
        raise ValueError("Louvain returned no communities.")

    fig = figure(
        title="Mean weighted degree per Louvain community (map partition)",
        width=950,
        height=480,
        x_range=stats["community"].tolist(),
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    source = ColumnDataSource(stats)
    fig.vbar(
        x="community",
        top="mean_degree",
        width=0.75,
        source=source,
        color="color",
        line_color="white",
    )
    fig.xaxis.major_label_orientation = 0.8
    fig.xaxis.axis_label = "Community"
    fig.yaxis.axis_label = "Mean weighted degree"
    return fig, stats


def plot_louvain_community_node_counts(
    graph_undirected: nx.Graph,
    communities: Iterable[set] | None = None,
    seed: int = 42,
    weight: str = "weight",
):
    stats = louvain_community_mean_degrees(
        graph_undirected,
        communities=communities,
        seed=seed,
        weight=weight,
    )
    if stats.empty:
        raise ValueError("Louvain returned no communities.")

    fig = figure(
        title="Number of nodes per Louvain community (map partition)",
        width=950,
        height=480,
        x_range=stats["community"].tolist(),
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    source = ColumnDataSource(stats)
    fig.vbar(
        x="community",
        top="n_nodes",
        width=0.75,
        source=source,
        color="color",
        line_color="white",
    )
    fig.xaxis.major_label_orientation = 0.8
    fig.xaxis.axis_label = "Community"
    fig.yaxis.axis_label = "Number of nodes"
    return fig, stats


def louvain_modularity_by_level(
    graph_undirected: nx.Graph,
    seed: int = 42,
    weight: str = "weight",
) -> pd.DataFrame:
    """Modularity from singleton start through each Louvain hierarchical level."""
    singleton_partition = [{node} for node in graph_undirected.nodes()]
    rows = [
        {
            "level": 0,
            "modularity": float(modularity(graph_undirected, singleton_partition, weight=weight)),
            "n_communities": graph_undirected.number_of_nodes(),
            "stage": "initial (singletons)",
        }
    ]
    final_partition = None
    for level, partition in enumerate(louvain_partitions(graph_undirected, seed=seed, weight=weight), start=1):
        final_partition = partition
        rows.append(
            {
                "level": level,
                "modularity": float(modularity(graph_undirected, partition, weight=weight)),
                "n_communities": len(partition),
                "stage": "hierarchical",
            }
        )
    if len(rows) > 1:
        if len(rows) >= 3:
            rows[-2]["stage"] = "pre-final (level before final)"
        rows[-1]["stage"] = "final (chosen partition)"

    if final_partition is not None:
        over_merged = [set(graph_undirected.nodes())]
        rows.append(
            {
                "level": rows[-1]["level"] + 1,
                "modularity": float(modularity(graph_undirected, over_merged, weight=weight)),
                "n_communities": 1,
                "stage": "next level (over-merged, worse Q)",
            }
        )
    return pd.DataFrame(rows)


def plot_louvain_modularity_trajectory(
    graph_undirected: nx.Graph,
    seed: int = 42,
    weight: str = "weight",
):
    trajectory = louvain_modularity_by_level(graph_undirected, seed=seed, weight=weight)
    if trajectory.empty:
        raise ValueError("Louvain produced no hierarchical levels.")

    initial_row = trajectory.iloc[0]
    final_rows = trajectory[trajectory["stage"] == "final (chosen partition)"]
    final_row = final_rows.iloc[0] if not final_rows.empty else trajectory.iloc[-2]
    pre_final_rows = trajectory[trajectory["stage"].str.contains("pre-final", na=False)]
    pre_final_row = pre_final_rows.iloc[0] if not pre_final_rows.empty else None
    over_rows = trajectory[trajectory["stage"].str.contains("over-merged", na=False)]
    over_row = over_rows.iloc[0] if not over_rows.empty else trajectory.iloc[-1]

    highlight_levels = [initial_row["level"], final_row["level"], over_row["level"]]
    highlight_mods = [initial_row["modularity"], final_row["modularity"], over_row["modularity"]]
    highlight_labels = ["Start (worst Q)", "Final (used)", "Next level (worse)"]
    highlight_colors = ["#d62728", "#2ca02c", "#ff7f0e"]
    if pre_final_row is not None:
        highlight_levels.insert(2, pre_final_row["level"])
        highlight_mods.insert(2, pre_final_row["modularity"])
        highlight_labels.insert(2, "Pre-final")
        highlight_colors.insert(2, "#9467bd")

    fig = figure(
        title="Louvain modularity: singletons → pre-final → final → over-merged",
        width=950,
        height=500,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    path_source = ColumnDataSource(trajectory)
    fig.line(x="level", y="modularity", line_width=2.5, color="#1f77b4", source=path_source)
    fig.circle(x="level", y="modularity", size=8, color="#9ecae1", source=path_source)

    highlights = ColumnDataSource(
        dict(
            level=highlight_levels,
            modularity=highlight_mods,
            label=highlight_labels,
            color=highlight_colors,
        )
    )
    fig.circle(
        x="level",
        y="modularity",
        size=14,
        color="color",
        source=highlights,
        legend_field="label",
    )
    fig.legend.location = "bottom_right"
    fig.legend.click_policy = "hide"

    annotations = LabelSet(
        x="level",
        y="modularity",
        text="label",
        y_offset=12,
        source=highlights,
        text_font_size="10pt",
    )
    fig.add_layout(annotations)

    fig.xaxis.axis_label = "Louvain level (0 = singletons, last = returned partition)"
    fig.yaxis.axis_label = "Modularity Q"
    return fig, trajectory


##### FIN ANALISIS DE PARTICIONES ###################################################################################

def _igraph_networkx_node_ids(ig_graph) -> List:
    """Map igraph vertices back to the original NetworkX node identifiers."""
    if "_nx_name" in ig_graph.vs.attributes():
        return list(ig_graph.vs["_nx_name"])
    if "name" in ig_graph.vs.attributes():
        return list(ig_graph.vs["name"])
    return list(range(ig_graph.vcount()))


def _membership_to_communities(node_ids: List, membership: Iterable[int]) -> List[set]:
    buckets: Dict[int, set] = {}
    for node_id, community_index in zip(node_ids, membership):
        buckets.setdefault(int(community_index), set()).add(node_id)
    return [members for members in buckets.values() if members]


def leiden_communities_plot(graph_undirected: nx.Graph, max_edges: int = 1200):
    if ig is not None:
        ig_graph = ig.Graph.from_networkx(graph_undirected)
        node_ids = _igraph_networkx_node_ids(ig_graph)
        leiden_kwargs = {"objective_function": "modularity"}
        partition_kwargs = {}
        if "weight" in ig_graph.es.attributes():
            leiden_kwargs["weights"] = "weight"
            partition_kwargs["weights"] = "weight"

        if hasattr(ig_graph, "community_leiden"):
            partition = ig_graph.community_leiden(**leiden_kwargs)
            communities = _membership_to_communities(node_ids, partition.membership)
            score = float(partition.modularity)
            return _plot_community_map(graph_undirected, communities, "Leiden Communities", score, max_edges=max_edges)

        if leidenalg is not None:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                **partition_kwargs,
            )
            communities = _membership_to_communities(node_ids, partition.membership)
            score = float(partition.modularity)
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


# ─── Dynamics: Random Walk ────────────────────────────────────────────────────

def simulate_random_walk(
    graph: nx.DiGraph,
    n_steps: int = 200_000,
    restart_prob: float = 0.15,
    seed: int = 42,
) -> Dict[int, float]:
    """Random walk with teleportation on a directed graph; returns normalised visit frequencies."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    adj = {v: list(graph.successors(v)) for v in nodes}
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    visit = np.zeros(n, dtype=np.int64)

    current = nodes[int(rng.integers(n))]
    for _ in range(n_steps):
        visit[node_to_idx[current]] += 1
        if rng.random() < restart_prob or not adj[current]:
            current = nodes[int(rng.integers(n))]
        else:
            nbrs = adj[current]
            current = nbrs[int(rng.integers(len(nbrs)))]

    freq = visit / n_steps
    return {v: float(freq[node_to_idx[v]]) for v in nodes}


def compute_mixing_time_tvd(
    graph: nx.DiGraph,
    start_node=None,
    damping: float = 0.85,
    epsilon: float = 0.05,
    max_steps: int = 25,
) -> Tuple[np.ndarray, int]:
    """
    Power-iterate the PageRank transition matrix starting from a delta distribution.
    Returns (tvd_per_step, first_step_where_tvd < epsilon).
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    T = np.zeros((n, n))
    for u in nodes:
        i = node_to_idx[u]
        succs = list(graph.successors(u))
        if succs:
            for v in succs:
                T[i, node_to_idx[v]] = 1.0 / len(succs)
        else:
            T[i, :] = 1.0 / n  # dangling node teleports uniformly

    PR = damping * T + (1.0 - damping) / n  # PageRank transition matrix

    pi_dict = nx.pagerank(graph, alpha=damping)
    pi = np.array([pi_dict[v] for v in nodes])

    if start_node is None:
        start_node = nodes[0]
    dist = np.zeros(n)
    dist[node_to_idx[start_node]] = 1.0

    tvd_series: List[float] = []
    mixing_time = max_steps
    for step in range(max_steps):
        tvd = 0.5 * float(np.sum(np.abs(dist - pi)))
        tvd_series.append(tvd)
        if tvd < epsilon and mixing_time == max_steps:
            mixing_time = step
        dist = dist @ PR

    return np.array(tvd_series), mixing_time


# ─── Dynamics: SIR Epidemic ───────────────────────────────────────────────────

def simulate_sir(
    graph: nx.Graph,
    beta: float,
    gamma: float,
    seed_node: int,
    n_steps: int = 60,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Discrete-time SIR epidemic on an undirected graph."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    adj = {v: list(graph.neighbors(v)) for v in nodes}

    state = {v: 0 for v in nodes}  # 0=S, 1=I, 2=R
    state[seed_node] = 1

    S_list, I_list, R_list = [n - 1], [1], [0]

    for _ in range(n_steps):
        new_I: set = set()
        new_R: set = set()
        for v in nodes:
            if state[v] == 1:
                for u in adj[v]:
                    if state[u] == 0 and rng.random() < beta:
                        new_I.add(u)
                if rng.random() < gamma:
                    new_R.add(v)
        for v in new_I:
            state[v] = 1
        for v in new_R:
            if state[v] == 1:
                state[v] = 2

        S = sum(1 for v in nodes if state[v] == 0)
        I = sum(1 for v in nodes if state[v] == 1)
        R = sum(1 for v in nodes if state[v] == 2)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        if I == 0:
            for _ in range(n_steps - len(S_list) + 1):
                S_list.append(S)
                I_list.append(0)
                R_list.append(R)
            break

    return {"S": S_list, "I": I_list, "R": R_list}


# ─── Dynamics: Visualisation ──────────────────────────────────────────────────

def plot_random_walk_frequency_map(
    graph: nx.DiGraph,
    n_steps: int = 200_000,
    max_edges: int = 700,
):
    """Stacked maps of simulated visit frequency and PageRank."""
    walk_freq = simulate_random_walk(graph, n_steps=n_steps)
    pr = nx.pagerank(graph, alpha=0.85)

    fig1 = _network_on_map(
        graph,
        title=f"Random Walk Visit Frequency ({n_steps:,} steps, restart p=0.15)",
        max_edges=max_edges,
        node_value=walk_freq,
        colorbar_title="Visit Freq",
        node_palette=Viridis256,
        node_size=9,
    )
    fig2 = _network_on_map(
        graph,
        title="PageRank — Analytical Stationary Distribution",
        max_edges=max_edges,
        node_value=pr,
        colorbar_title="PageRank",
        node_palette=Viridis256,
        node_size=9,
    )
    return column(fig1, fig2)


def plot_mixing_time_convergence(
    graph: nx.DiGraph,
    damping: float = 0.85,
    epsilon: float = 0.05,
    max_steps: int = 20,
):
    """TVD to stationary vs step, for three different starting nodes."""
    nodes = list(graph.nodes())
    n = len(nodes)
    degrees_out = dict(graph.out_degree())
    sorted_by_out = sorted(nodes, key=lambda v: degrees_out[v])

    start_configs = [
        ("Highest out-degree", sorted_by_out[-1], "#d62728"),
        ("Median out-degree", sorted_by_out[n // 2], "#1f77b4"),
        ("Lowest out-degree", sorted_by_out[0], "#2ca02c"),
    ]

    fig = figure(
        title="Random Walk Convergence: TVD to Stationary Distribution",
        width=950,
        height=420,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    max_len = 0
    for label, node, color in start_configs:
        tvd, mt = compute_mixing_time_tvd(
            graph, start_node=node, damping=damping, epsilon=epsilon, max_steps=max_steps
        )
        name = graph.nodes[node].get("name", str(node))
        steps = list(range(len(tvd)))
        max_len = max(max_len, len(tvd))
        fig.line(
            steps,
            tvd.tolist(),
            color=color,
            line_width=2.5,
            legend_label=f"{label} — {name} (mixes at step {mt})",
        )

    fig.line(
        [0, max_len - 1],
        [epsilon, epsilon],
        line_dash="dashed",
        color="black",
        line_width=1.5,
        legend_label=f"ε = {epsilon}",
    )
    fig.xaxis.axis_label = "Step"
    fig.yaxis.axis_label = "Total Variation Distance"
    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    return fig


def plot_sir_epidemics(
    graph: nx.Graph,
    beta: float = 0.003,
    gamma: float = 0.10,
    n_steps: int = 60,
    seed: int = 42,
):
    """SIR epidemic curves starting from the top hub, a median, and the most peripheral node."""
    nodes = list(graph.nodes())
    n = len(nodes)
    degrees = dict(graph.degree())
    sorted_by_deg = sorted(nodes, key=lambda v: degrees[v])

    configs = [
        (sorted_by_deg[-1], "#d62728"),
        (sorted_by_deg[n // 2], "#1f77b4"),
        (sorted_by_deg[0], "#2ca02c"),
    ]

    steps = list(range(n_steps + 1))

    fig_i = figure(
        title=f"SIR Epidemic — Infected fraction  (β={beta}, γ={gamma})",
        width=950,
        height=400,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )
    fig_r = figure(
        title=f"SIR Epidemic — Recovered fraction  (β={beta}, γ={gamma})",
        width=950,
        height=400,
        tools="pan,wheel_zoom,box_zoom,save,reset",
    )

    for node, color in configs:
        name = graph.nodes[node].get("name", str(node))
        k = degrees[node]
        label = f"{name} (k={k})"
        res = simulate_sir(graph, beta=beta, gamma=gamma, seed_node=node, n_steps=n_steps, seed=seed)
        i_frac = [v / n for v in res["I"]]
        r_frac = [v / n for v in res["R"]]
        xs = steps[: len(i_frac)]
        fig_i.line(xs, i_frac, color=color, line_width=2.5, legend_label=label)
        fig_r.line(steps[: len(r_frac)], r_frac, color=color, line_width=2.5, legend_label=label)

    for f in (fig_i, fig_r):
        f.xaxis.axis_label = "Time Step"
        f.yaxis.axis_label = "Fraction of Population"
        f.legend.location = "top_right"
        f.legend.click_policy = "hide"

    return column(fig_i, fig_r)


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
