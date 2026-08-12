# SIMI Paper Workspace (Independent from `cio_pipeline-2`)

Working title: **Systemically Important Material Inputs: Measuring Shock Propagation and Systemic Centrality in a Developing Economy's Construction Cost Structure**.

This folder is standalone for Paper 4 drafting and estimation, while consuming `panel_v1.0` from `cio_pipeline-2`.

## What is implemented

1. `src/volatility/spillover.py`
   - generalized FEVD (Pesaran-Shin) and Diebold-Yilmaz spillover table
   - transmission scores (`to_others`, `from_others`, `net`)
2. `src/network/`
   - correlation networks (`pearson`, `spearman`, `partial`)
   - Granger-causality network constructor
   - centrality table (eigenvector, betweenness, PageRank)
3. `scripts/run_simi_pipeline.py`
   - computes SIMI under 3 weighting schemes
   - writes tables and figures (spillover heatmap + MST SIMI map)
4. `scripts/run_robustness.py`
   - threshold × window × weighting × subsample rank-stability sweep
   - writes a single stability heatmap
5. `papers/P4/manuscript.md`
   - full draft skeleton with methods, hypotheses, robustness, and practitioner interpretation.

## Usage

Run baseline:

```bash
python scripts/run_simi_pipeline.py
```

Run robustness:

```bash
python scripts/run_robustness.py
```

Outputs are written into `papers/P4/tables` and `papers/P4/figures`.

## Notes on CIPI basket shares

The code estimates non-negative CIPI material basket-share proxies directly from the published `CIPI_MAT` series using constrained least squares (`nnls`). This keeps economic significance anchored to the official CIPI structure while avoiding ad hoc BOQ weights.
