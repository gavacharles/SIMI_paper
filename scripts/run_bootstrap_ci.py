from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.network.centrality import centrality_table
from src.network.estimation import build_correlation_network, correlation_matrix
from src.simi.scoring import compute_simi, estimate_cipi_basket_shares, weighting_scenarios
from src.volatility.spillover import (
    SpilloverConfig,
    fit_spillover_table,
    identify_material_columns,
    log_returns,
    prepare_returns,
    transmission_scores,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def moving_block_bootstrap(data: pd.DataFrame, block_len: int, rng: np.random.Generator) -> pd.DataFrame:
    n = len(data)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
    out = data.iloc[idx].reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Moving-block bootstrap CIs for SIMI scores.")
    p.add_argument(
        "--panel-path",
        default=str((ROOT.parents[1] / "cio_pipeline-2" / "data" / "processed" / "panel_v1.0.csv").resolve()),
    )
    p.add_argument("--output-dir", default=str((ROOT / "papers" / "P4_final").resolve()))
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--block-len", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--energy-proxy", default="ppi_manufacturing")
    p.add_argument("--extra-nodes", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    clean = prepare_returns(returns, min_obs)
    shares = estimate_cipi_basket_shares(panel, [m for m in materials if m in panel.columns])
    weights = weighting_scenarios()["equal"]

    rng = np.random.default_rng(args.seed)
    rows = []
    failures = 0
    for b in range(args.n_boot):
        try:
            resampled = moving_block_bootstrap(clean, args.block_len, rng)
            spill = fit_spillover_table(resampled, SpilloverConfig(lags=2, horizon=10, min_obs=min_obs))
            tx = transmission_scores(spill)
            corr = correlation_matrix(resampled, correlation_type="partial")
            net = build_correlation_network(corr, threshold=0.2)
            cent = centrality_table(net)
            simi = compute_simi(tx["to_others"], cent["centrality_composite"], shares, weights)
            tmp = simi[["simi_score", "rank"]].copy()
            tmp["boot_id"] = b
            tmp = tmp.reset_index(names="node")
            rows.append(tmp)
        except Exception as e:  # noqa: BLE001
            failures += 1
            continue

    if not rows:
        raise RuntimeError("All bootstrap replications failed.")

    boot = pd.concat(rows, ignore_index=True)
    boot.to_csv(out_dir / "tables" / "bootstrap_simi_draws.csv", index=False)

    summary = (
        boot.groupby("node")["simi_score"]
        .agg(["mean", "std", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)])
        .rename(columns={"<lambda_0>": "ci_lo_2.5", "<lambda_1>": "ci_hi_97.5"})
    )
    rank_summary = boot.groupby("node")["rank"].agg(["mean", "std", "min", "max"]).rename(
        columns={"mean": "rank_mean", "std": "rank_sd", "min": "rank_best", "max": "rank_worst"}
    )
    out = summary.join(rank_summary).sort_values("mean", ascending=False)
    out.to_csv(out_dir / "tables" / "bootstrap_simi_summary.csv")

    n_ok = args.n_boot - failures
    print(f"Bootstrap complete: {n_ok}/{args.n_boot} replications succeeded, block_len={args.block_len}.")
    print(out.head(8).to_string())

    top = out.index[0]
    second = out.index[1]
    p_top_beats_second = float((boot.pivot(index="boot_id", columns="node", values="simi_score")[top] >
                                 boot.pivot(index="boot_id", columns="node", values="simi_score")[second]).mean())
    print(f"\nP({top} SIMI score > {second} SIMI score) across bootstrap draws: {p_top_beats_second:.3f}")


if __name__ == "__main__":
    main()
