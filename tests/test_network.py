from __future__ import annotations

import networkx as nx

from src.network.centrality import centrality_table


def test_pagerank_does_not_crash_on_negative_weights() -> None:
    """
    Regression test: correlation-network edges carry a signed `weight`
    (raw Pearson/partial correlation). PageRank requires non-negative
    weights to form a valid stochastic transition matrix; feeding it signed
    weights directly used to throw PowerIterationFailedConvergence whenever
    a graph had a strongly negative edge. centrality_table must route
    PageRank/eigenvector through `abs_weight` instead.
    """
    g = nx.Graph()
    g.add_edge("a", "b", weight=-0.9, abs_weight=0.9, distance=0.1)
    g.add_edge("b", "c", weight=0.6, abs_weight=0.6, distance=0.4)
    g.add_edge("a", "c", weight=-0.3, abs_weight=0.3, distance=0.7)

    out = centrality_table(g)

    assert set(out.index) == {"a", "b", "c"}
    assert out["pagerank"].notna().all()
    assert (out["pagerank"] >= 0).all()


def test_centrality_uses_strength_not_signed_weight() -> None:
    """
    A node connected only by a strong *negative* correlation should be
    treated as highly central (strongly linked), not weakly central, since
    strength is |correlation|. Using signed weight would under-rank it.
    """
    g = nx.Graph()
    g.add_edge("hub", "x", weight=-0.95, abs_weight=0.95, distance=0.05)
    g.add_edge("hub", "y", weight=0.95, abs_weight=0.95, distance=0.05)
    g.add_node("isolated")

    out = centrality_table(g)

    assert out.loc["hub", "centrality_composite"] > out.loc["isolated", "centrality_composite"]
