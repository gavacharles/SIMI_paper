from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.network.centrality import centrality_table
from src.network.estimation import (
    build_correlation_network,
    build_spillover_network,
    correlation_matrix,
    minimum_spanning_tree,
)
from src.network.plots import plot_mst_simi, plot_spillover_heatmap
from src.simi.scoring import compute_simi, estimate_cipi_basket_shares, weighting_scenarios
from src.volatility.spillover import (
    SpilloverConfig,
    fit_spillover_table,
    identify_material_columns,
    log_returns,
    prepare_returns,
    transmission_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SIMI paper pipeline from CiO panel.")
    p.add_argument(
        "--panel-path",
        default=str((ROOT.parents[1] / "cio_pipeline-2" / "data" / "processed" / "panel_v1.0.csv").resolve()),
    )
    p.add_argument("--output-dir", default=str((ROOT / "papers" / "P4").resolve()))
    p.add_argument("--lags", type=int, default=2)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--corr-type", choices=["pearson", "spearman", "partial"], default="partial")
    p.add_argument("--corr-threshold", type=float, default=0.2)
    p.add_argument("--spill-threshold", type=float, default=2.0)
    p.add_argument("--energy-proxy", default="ppi_manufacturing")
    p.add_argument("--extra-nodes", default="", help="Comma-separated extra columns to include as auxiliary nodes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel_path, parse_dates=["date"]).sort_values("date")

    materials = identify_material_columns(panel.columns)
    aux_nodes = ["exchange_rate"]
    if args.energy_proxy in panel.columns:
        aux_nodes.append(args.energy_proxy)
    if args.extra_nodes:
        aux_nodes.extend(n.strip() for n in args.extra_nodes.split(",") if n.strip())

    node_cols = [c for c in materials + aux_nodes if c in panel.columns]
    returns = log_returns(panel, node_cols)

    if args.window > 0 and len(returns) > args.window:
        returns = returns.tail(args.window)

    min_obs = max(36, args.window // 2)
    spill_cfg = SpilloverConfig(lags=args.lags, horizon=args.horizon, min_obs=min_obs)
    clean_returns = prepare_returns(returns, min_obs)
    spill = fit_spillover_table(clean_returns, spill_cfg)
    spill.to_csv(out_dir / "tables" / "spillover_table.csv")

    tx = transmission_scores(spill)
    tx.to_csv(out_dir / "tables" / "spillover_transmission.csv")

    corr = correlation_matrix(clean_returns, correlation_type=args.corr_type)
    corr.to_csv(out_dir / "tables" / "correlation_matrix.csv")

    corr_net = build_correlation_network(corr, threshold=args.corr_threshold)
    spill_net = build_spillover_network(spill, threshold=args.spill_threshold)
    nx.write_graphml(spill_net, out_dir / "tables" / "spillover_network.graphml")
    nx.write_graphml(corr_net, out_dir / "tables" / "correlation_network.graphml")

    cent = centrality_table(corr_net)
    cent.to_csv(out_dir / "tables" / "centrality_table.csv")

    material_shares = estimate_cipi_basket_shares(panel, [m for m in materials if m in panel.columns])
    material_shares.to_csv(out_dir / "tables" / "cipi_basket_share_proxy.csv", header=["basket_share"])

    rows = []
    for name, w in weighting_scenarios().items():
        simi = compute_simi(
            spill_transmission=tx["to_others"],
            centrality_composite=cent["centrality_composite"],
            basket_shares=material_shares,
            weights=w,
        )
        simi["scenario"] = name
        rows.append(simi.reset_index(names="node"))
        if name == "equal":
            simi.to_csv(out_dir / "tables" / "simi_ranking_equal.csv")

    simi_long = pd.concat(rows, ignore_index=True)
    simi_long.to_csv(out_dir / "tables" / "simi_rankings_all_scenarios.csv", index=False)

    mst = minimum_spanning_tree(corr.loc[corr.index.intersection(material_shares.index), corr.columns.intersection(material_shares.index)])
    simi_equal = pd.read_csv(out_dir / "tables" / "simi_ranking_equal.csv", index_col=0)

    plot_spillover_heatmap(spill, str(out_dir / "figures" / "spillover_heatmap.png"))
    plot_mst_simi(mst, simi_equal["simi_score"], str(out_dir / "figures" / "mst_simi_map.png"))

    print("SIMI pipeline complete.")
    print("Top-5 (equal weighting):")
    print(simi_equal.head(5)[["simi_score", "rank"]].to_string())


if __name__ == "__main__":
    main()
