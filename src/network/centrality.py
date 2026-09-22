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

    # Correlation-network edges carry a *signed* `weight` (raw Pearson/partial
    # correlation). Eigenvector centrality and PageRank treat "weight" as
    # connection *strength* and require it to be non-negative to be
    # well-defined (PageRank needs it to form a valid stochastic transition
    # matrix -- it will fail to converge, as it did here once a strongly
    # negative edge was introduced, if given signed weights). Betweenness
    # instead treats "weight" as a shortest-path *distance*, where a raw
    # correlation is the wrong direction entirely (higher correlation should
    # mean "closer", not "farther"). `build_correlation_network` already
    # provides the right attribute for each: `abs_weight` (strength) and
    # `distance = 1 - abs_weight` (used the same way as in the MST). Fall
    # back to the signed `weight` only for graphs that don't carry these
    # (e.g. a spillover network, whose weights are already non-negative).
    has_edges = h.number_of_edges() > 0
    sample_attrs = next(iter(h.edges(data=True)))[2] if has_edges else {}
    strength_attr = "abs_weight" if "abs_weight" in sample_attrs else "weight"
    distance_attr = "distance" if "distance" in sample_attrs else "weight"

    eig = nx.eigenvector_centrality_numpy(h, weight=strength_attr) if has_edges else {n: 0.0 for n in h.nodes}
    btw = nx.betweenness_centrality(h, weight=distance_attr, normalized=True) if has_edges else {n: 0.0 for n in h.nodes}
    pr = nx.pagerank(h, weight=strength_attr) if has_edges else {n: 0.0 for n in h.nodes}

    out = pd.DataFrame({"eigenvector": eig, "betweenness": btw, "pagerank": pr}).fillna(0.0)
    out["centrality_composite"] = out.mean(axis=1)
    return out.sort_values("centrality_composite", ascending=False)
