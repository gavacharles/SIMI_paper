from __future__ import annotations

import networkx as nx
import pandas as pd


def centrality_table(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["eigenvector", "betweenness", "pagerank", "centrality_composite"])

    if isinstance(graph, nx.DiGraph):
        h = graph.copy()
    else:
        h = graph.to_directed()

    eig = nx.eigenvector_centrality_numpy(h, weight="weight") if h.number_of_edges() > 0 else {n: 0.0 for n in h.nodes}
    btw = nx.betweenness_centrality(h, weight="weight", normalized=True) if h.number_of_edges() > 0 else {n: 0.0 for n in h.nodes}
    pr = nx.pagerank(h, weight="weight") if h.number_of_edges() > 0 else {n: 0.0 for n in h.nodes}

    out = pd.DataFrame({"eigenvector": eig, "betweenness": btw, "pagerank": pr}).fillna(0.0)
    out["centrality_composite"] = out.mean(axis=1)
    return out.sort_values("centrality_composite", ascending=False)
