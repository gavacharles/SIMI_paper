from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.network.centrality import centrality_table
from src.network.estimation import build_correlation_network, correlation_matrix
from src.simi.scoring import compute_simi, estimate_cipi_basket_shares, weighting_scenarios
from src.volatility.spillover import SpilloverConfig, fit_spillover_table, identify_material_columns, log_returns, transmission_scores


SUBSAMPLES = {
    "pre_covid": ("2016-01-01", "2019-12-31"),
    "covid": ("2020-01-01", "2021-12-31"),
    "post_2022": ("2022-01-01", "2026-12-31"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SIMI robustness sweep.")
    p.add_argument(
        "--panel-path",
        default=str((ROOT.parents[1] / "cio_pipeline-2" / "data" / "processed" / "panel_v1.0.csv").resolve()),
    )
    p.add_argument("--output-dir", default=str((ROOT / "papers" / "P4").resolve()))
    p.add_argument("--corr-type", choices=["pearson", "spearman", "partial"], default="partial")
    p.add_argument("--energy-proxy", default="ppi_manufacturing")
    return p.parse_args()


def run_scenario(
    panel: pd.DataFrame,
    materials: list[str],
    window: int,
    threshold: float,
    corr_type: str,
    scenario: str,
    energy_proxy: str,
) -> pd.DataFrame:
    aux = ["exchange_rate"]
    if energy_proxy in panel.columns:
        aux.append(energy_proxy)

    cols = [c for c in materials + aux if c in panel.columns]
    ret = log_returns(panel, cols)
    if len(ret) > window:
        ret = ret.tail(window)

    spill = fit_spillover_table(ret, SpilloverConfig(lags=2, horizon=10, min_obs=max(24, window // 2)))
    tx = transmission_scores(spill)

    corr = correlation_matrix(ret, correlation_type=corr_type)
    net = build_correlation_network(corr, threshold=threshold)
    cent = centrality_table(net)

    shares = estimate_cipi_basket_shares(panel, [m for m in materials if m in panel.columns])
    w = weighting_scenarios()[scenario]
    simi = compute_simi(tx["to_others"], cent["centrality_composite"], shares, w)
    return simi


def plot_rank_stability(stability: pd.DataFrame, out_path: Path) -> None:
    mat = stability.pivot(index="node", columns="scenario_id", values="rank")
    mat = mat.sort_values(mat.columns.tolist()[0])

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_title("SIMI rank stability across thresholds, windows, and weightings")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Rank (1 = highest systemic importance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel_path, parse_dates=["date"]).sort_values("date")
    materials = identify_material_columns(panel.columns)

    thresholds = [0.15, 0.20, 0.25, 0.30]
    windows = [36, 48, 60]
    weightings = list(weighting_scenarios().keys())

    rows = []
    for sample_name, (start, end) in SUBSAMPLES.items():
        sub = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
        if len(sub) < 30:
            continue

        for window, thr, wname in itertools.product(windows, thresholds, weightings):
            try:
                simi = run_scenario(sub, materials, window, thr, args.corr_type, wname, args.energy_proxy)
                scenario_id = f"{sample_name}|w{window}|t{thr:.2f}|{wname}"
                tmp = simi[["rank"]].copy().reset_index(names="node")
                tmp["scenario_id"] = scenario_id
                tmp["window"] = window
                tmp["threshold"] = thr
                tmp["weighting"] = wname
                tmp["subsample"] = sample_name
                rows.append(tmp)
            except Exception:
                continue

    if not rows:
        raise RuntimeError("No robustness scenarios completed. Check panel coverage.")

    stability = pd.concat(rows, ignore_index=True)
    stability.to_csv(out_dir / "tables" / "rank_stability_long.csv", index=False)

    summary = (
        stability.groupby("node", as_index=False)["rank"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "rank_mean", "std": "rank_sd", "min": "rank_best", "max": "rank_worst"})
        .sort_values("rank_mean")
    )
    summary.to_csv(out_dir / "tables" / "rank_stability_summary.csv", index=False)

    plot_rank_stability(stability, out_dir / "figures" / "rank_stability_heatmap.png")
    print("Robustness sweep complete.")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
