from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_spillover_heatmap(spill: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(spill.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(spill.columns)))
    ax.set_xticklabels(spill.columns, rotation=90)
    ax.set_yticks(range(len(spill.index)))
    ax.set_yticklabels(spill.index)
    ax.set_title("Generalized FEVD spillover matrix (%)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_mst_simi(mst: nx.Graph, simi_scores: pd.Series, out_path: str) -> None:
    if mst.number_of_nodes() == 0:
        return

    score = simi_scores.reindex(list(mst.nodes())).fillna(simi_scores.median() if len(simi_scores) else 1.0)
    node_size = 800 + 4200 * (score - score.min()) / (score.max() - score.min() + 1e-9)

    pos = nx.spring_layout(mst, seed=42)
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1.6, ax=ax)
    nodes = nx.draw_networkx_nodes(
        mst,
        pos,
        node_size=node_size,
        node_color=score.values,
        cmap="viridis",
        ax=ax,
    )
    nx.draw_networkx_labels(mst, pos, font_size=9, ax=ax)
    cbar = fig.colorbar(nodes, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("SIMI score")
    ax.set_title("Minimum spanning tree with SIMI-scaled node size")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
