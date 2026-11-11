#!/usr/bin/env python3
"""Parameter and belief recovery for the double-map simulations.

Run: python parameter_recovery_analysis.py
Requires numpy, pandas, scipy, matplotlib. Optional choice metrics require arviz.
Errors = fitted posterior mean minus generating parameter.
Coverage is descriptive for the simulation grid, not formal SBC.
Beliefs are plug-in trajectories at posterior-mean hazards (no credible bands).
Correlation is undefined for constant vectors and is saved as NaN.
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

path = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model"
data_path = os.path.join(path, "simulation_results_balanced100_fixedbeta")
fit_path = os.path.join(path, "fitting_results", "parameter_recovery100_fixedbeta")
model_parameters = {
    "flat_NoDecay": ["alpha", "beta"],
    "flat_Decay": ["beta", "K"],
    "hybrid_FreeWeights_NoDecay": ["alpha", "h_map", "h_peak", "w_mf", "w_map"],
    "hybrid_FreeWeights_Decay": ["h_map", "h_peak", "w_mf", "w_map", "K"],
    "hybrid_Pearce_Hall_NoDecay": ["alpha", "h_map", "h_peak", "eta"],
    "hybrid_Pearce_Hall_Decay": ["h_map", "h_peak", "eta", "K"],
    "full_Bayesian": ["beta", "h_map", "h_peak"],
}
BAYES_MODELS = {m for m in model_parameters if m.startswith("hybrid_") or m == "full_Bayesian"}
MAX_RHAT, MIN_ESS_BULK, MIN_ESS_TAIL = 1.01, 400, 400


def numeric(row, key):
    try:
        return float(row.get(key, np.nan))
    except (ValueError, TypeError):
        return np.nan


def correlation(x, y, method="pearson"):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float((pearsonr if method == "pearson" else spearmanr)(x, y).statistic)


def finite_mean(x):
    x = np.asarray(x, float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def hdi_columns(frame):
    cols = sorted([c for c in frame if c.startswith("hdi_") and c.endswith("%")],
                  key=lambda c: float(c[4:-1]))
    if len(cols) != 2:
        raise ValueError("Expected two hdi_*% summary columns")
    return cols[0], cols[1], (float(cols[1][4:-1]) - float(cols[0][4:-1])) / 100


def distribution_distances(p, q):
    # Each row is one probability distribution, summing over all state axes.
    axes = tuple(range(1, p.ndim))
    tv = 0.5 * np.abs(p - q).sum(axis=axes)
    mid = 0.5 * (p + q)
    eps = 1e-300
    js = 0.5 * (p * np.log(np.maximum(p, eps) / np.maximum(mid, eps)) +
                q * np.log(np.maximum(q, eps) / np.maximum(mid, eps))).sum(axis=axes)
    return tv, js


def validate_beliefs(x, label):
    if x.ndim != 3 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{label}: invalid trial x map x peak probability array")
    if not np.allclose(x.sum(axis=(1, 2)), 1, atol=1e-6):
        raise ValueError(f"{label}: probabilities do not sum to one")


def analyze_beliefs(data, belief_file, model_type, parameter_set, window):
    metrics, trials, events = [], [], []
    same_block = data.blockID.eq(data.blockID.shift()).to_numpy()
    with np.load(belief_file, allow_pickle=False) as saved:
        if str(saved["model_type"].item()) != model_type or int(saved["parameter_set"].item()) != parameter_set:
            raise ValueError("Belief NPZ model/parameter-set identity mismatch")
        if not np.array_equal(saved["blockID"], data.blockID.to_numpy()):
            raise ValueError("Belief NPZ/CSV blockID order mismatch")
        for stage in ("joint_prior", "joint_posterior"):
            p = np.stack([np.asarray(json.loads(v), float) for v in data[stage]])
            q = np.asarray(saved[stage], float)
            validate_beliefs(p, "true " + stage)
            validate_beliefs(q, "fitted " + stage)
            if p.shape != q.shape or len(p) != len(data):
                raise ValueError(f"{stage}: true and fitted belief shapes disagree")
            tv, js = distribution_distances(p, q)
            base = dict(model_type=model_type, parameter_set=parameter_set, stage=stage)
            # Separate joint, map-marginal, and peak-marginal distances.
            for target, a, b in (("joint", p, q), ("map", p.sum(axis=2), q.sum(axis=2)),
                                 ("peak", p.sum(axis=1), q.sum(axis=1))):
                dtv, djs = distribution_distances(a, b)
                metrics.append(dict(base, target=target, state_index=np.nan, n_trials=len(p),
                                    pearson_r=np.nan, spearman_r=np.nan,
                                    mae=float(np.abs(a-b).mean()), rmse=float(np.sqrt(np.mean((a-b)**2))),
                                    mean_total_variation=float(dtv.mean()), mean_js_nats=float(djs.mean()),
                                    change_mae=np.nan, change_pearson_r=np.nan))
                if target == "joint":
                    continue
                # Correlate each state's probability through time; never numeric peak labels.
                for state in range(a.shape[1]):
                    x, y = a[:, state], b[:, state]
                    dx, dy = np.diff(x)[same_block[1:]], np.diff(y)[same_block[1:]]
                    metrics.append(dict(base, target=target+"_state", state_index=state,
                                        n_trials=len(x), pearson_r=correlation(x, y),
                                        spearman_r=correlation(x, y, "spearman"),
                                        mae=float(np.abs(x-y).mean()), rmse=float(np.sqrt(np.mean((x-y)**2))),
                                        mean_total_variation=np.nan, mean_js_nats=np.nan,
                                        change_mae=finite_mean(np.abs(dx-dy)), change_pearson_r=correlation(dx, dy)))
            for t in range(len(data)):
                trials.append(dict(base, row_index=t, blockID=data.blockID.iloc[t],
                                   trialID=data.trialID.iloc[t], total_variation=tv[t], js_nats=js[t]))
            map_change = data.ring_label.ne(data.ring_label.shift()).to_numpy() & same_block
            peak_change = data.peak_idx.ne(data.peak_idx.shift()).to_numpy() & same_block
            masks = {"map_change": map_change, "peak_change": peak_change,
                     "peak_only_change": peak_change & ~map_change}
            for event_type, mask in masks.items():
                event_ids = np.flatnonzero(mask)
                for t in event_ids:
                    for offset in range(-window, window + 1):
                        j = t + offset
                        if 0 <= j < len(data) and data.blockID.iloc[j] == data.blockID.iloc[t]:
                            # Each row is event-relative; windows can overlap.
                            events.append(dict(base, event_type=event_type, event_row_index=int(t),
                                               position=offset, row_index=j, total_variation=tv[j], js_nats=js[j]))
    return metrics, trials, events


def choice_metrics(posterior_file, data):
    import arviz as az
    idata = az.from_netcdf(posterior_file)
    try:
        logp = np.asarray(idata.log_likelihood["choice"] if "log_likelihood" in idata.groups()
                          else idata.posterior["trial_loglik"], float)
        if logp.ndim != 3 or logp.shape[-1] != len(data):
            raise ValueError("Expected chain x draw x trial log likelihood")
        choices = data.choice.to_numpy()
        if not np.isin(choices, [0, 1]).all():
            raise ValueError("Choice coding must be 0=left, 1=right")
        chosen = np.clip(np.exp(logp).mean(axis=(0, 1)), 1e-300, 1)
        left = choices == 0
        prob_left = np.where(left, chosen, 1-chosen)
        return dict(choice_accuracy=float(np.mean((prob_left >= .5) == left)),
                    choice_log_loss=float(-np.log(chosen).mean()),
                    choice_brier_score=float(np.mean((prob_left-left)**2)))
    finally:
        idata.close()


def summarize_parameters(estimates):
    rows = []
    if estimates.empty:
        return pd.DataFrame()
    for (model_type, parameter), g in estimates.groupby(["model_type", "parameter"], sort=False):
        valid = g[np.isfinite(g.true_value) & np.isfinite(g.estimated_mean)]
        x, y = valid.true_value.to_numpy(), valid.estimated_mean.to_numpy()
        e = y-x
        rows.append(dict(model_type=model_type, parameter=parameter, n_fits=len(g), n_valid=len(valid),
                         n_distinct_true_values=len(np.unique(x)), pearson_r=correlation(x, y),
                         spearman_r=correlation(x, y, "spearman"), bias=finite_mean(e),
                         mae=finite_mean(np.abs(e)), rmse=np.sqrt(finite_mean(e**2)),
                         median_absolute_error=float(np.median(np.abs(e))) if len(e) else np.nan,
                         hdi_coverage=finite_mean(g.hdi_contains_true),
                         mean_hdi_width=finite_mean(g.hdi_width),
                         mean_hdi_probability=finite_mean(g.hdi_probability),
                         mean_posterior_sd=finite_mean(g.estimated_sd)))
    return pd.DataFrame(rows)


def save_plots(estimates, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (model_type, parameter), g in estimates.groupby(["model_type", "parameter"], sort=False):
        g = g[np.isfinite(g.true_value) & np.isfinite(g.estimated_mean)]
        if g.empty:
            continue
        folder = os.path.join(output_dir, "plots", model_type)
        os.makedirs(folder, exist_ok=True)
        x, y = g.true_value.to_numpy(), g.estimated_mean.to_numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        # Draw HDIs by endpoints: a posterior mean need not be inside the HDI.
        for row in g.itertuples():
            if np.isfinite(row.hdi_lower) and np.isfinite(row.hdi_upper):
                ax.vlines(row.true_value, row.hdi_lower, row.hdi_upper, color="tab:blue", alpha=.15, linewidth=1)
        ax.scatter(x, y, s=24, alpha=.65)
        limits = np.concatenate([x, y, g.hdi_lower.to_numpy(), g.hdi_upper.to_numpy()])
        limits = limits[np.isfinite(limits)]
        lo, hi = limits.min(), limits.max()
        pad = max((hi-lo)*.06, .001)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="black", linewidth=1)
        ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
               xlabel="True parameter", ylabel="Fitted posterior mean", title=f"{model_type}\n{parameter}")
        ax.set_aspect("equal", adjustable="box")
        r = correlation(x, y)
        rtext = f"{r:.3f}" if np.isfinite(r) else "undefined"
        ax.text(.04, .96, f"n = {len(g)}\nr = {rtext}\nMAE = {np.abs(y-x).mean():.4g}",
                transform=ax.transAxes, va="top")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(folder, f"{parameter}.{ext}"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=data_path)
    parser.add_argument("--fit-path", default=fit_path)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", action="append", choices=list(model_parameters))
    parser.add_argument("--summary-only", action="store_true", help="Skip beliefs and choice metrics")
    parser.add_argument("--with-choice-metrics", action="store_true", help="Read NetCDF; in-sample choice scores only")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--event-window", type=int, default=5)
    args = parser.parse_args()
    if args.event_window < 0:
        parser.error("--event-window must be nonnegative")
    output_dir = args.output_dir or os.path.join(args.fit_path, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    diagnostics, estimates, beliefs, trial_beliefs, event_beliefs = [], [], [], [], []
    for model_type in args.model or list(model_parameters):
        files = sorted(glob.glob(os.path.join(args.data_path, model_type, "double_map",
                                             f"{model_type}_double_parameter_set_*.csv")))
        print(f"{model_type}: {len(files)} datasets", flush=True)
        for csv_file in files:
            stem = os.path.splitext(os.path.basename(csv_file))[0]
            parameter_set = int(stem.rsplit("_", 1)[1])
            base = dict(model_type=model_type, parameter_set=parameter_set)
            fit_dir = os.path.join(args.fit_path, model_type)
            summary_file = os.path.join(fit_dir, stem+"_summary.csv")
            posterior_file = os.path.join(fit_dir, stem+"_posterior.nc")
            belief_file = os.path.join(fit_dir, "beliefs", f"parameter_set_{parameter_set:06d}_beliefs.npz")
            meta_file = os.path.join(args.data_path, model_type, f"{model_type}_parameter_set_{parameter_set:06d}_metadata.json")
            diag = dict(base, has_summary=os.path.isfile(summary_file), has_posterior=os.path.isfile(posterior_file),
                        has_beliefs=os.path.isfile(belief_file), parameters_analyzed=False, beliefs_analyzed=False,
                        max_r_hat=np.nan, min_ess_bulk=np.nan, min_ess_tail=np.nan, n_divergences=np.nan,
                        divergence_rate=np.nan, flagged_convergence=np.nan, diagnostics_complete=False)
            issues = []
            try:
                summary = pd.read_csv(summary_file, index_col=0)
                with open(meta_file, encoding="utf-8") as f:
                    metadata = json.load(f)
                if metadata["model_type"] != model_type or int(metadata["parameter_set"]) != parameter_set:
                    raise ValueError("Metadata identity mismatch")
                low, high, probability = hdi_columns(summary)
                local = []
                for parameter in model_parameters[model_type]:
                    row = summary.loc[parameter]
                    if "model_type" in row and row["model_type"] != model_type:
                        raise ValueError("Summary model identity mismatch")
                    if "parameter_set" in row and int(row["parameter_set"]) != parameter_set:
                        raise ValueError("Summary parameter-set identity mismatch")
                    true, mean = float(metadata["parameters"][parameter]), numeric(row, "mean")
                    lower, upper = numeric(row, low), numeric(row, high)
                    interval_valid = np.isfinite([lower, upper, true]).all() and lower <= upper
                    local.append(dict(base, parameter=parameter, true_value=true, estimated_mean=mean,
                                      estimated_sd=numeric(row, "sd"), error=mean-true, absolute_error=abs(mean-true),
                                      hdi_lower=lower, hdi_upper=upper, hdi_width=upper-lower if interval_valid else np.nan,
                                      hdi_probability=probability,
                                      hdi_contains_true=float(lower <= true <= upper) if interval_valid else np.nan,
                                      r_hat=numeric(row, "r_hat"), ess_bulk=numeric(row, "ess_bulk"),
                                      ess_tail=numeric(row, "ess_tail")))
                estimates.extend(local)
                diag["parameters_analyzed"] = True
                for source, target, reducer in (("r_hat", "max_r_hat", np.max), ("ess_bulk", "min_ess_bulk", np.min),
                                                 ("ess_tail", "min_ess_tail", np.min)):
                    v = np.asarray([r[source] for r in local])
                    diag[target] = float(reducer(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan
                for col in ("n_divergences", "divergence_rate"):
                    if col in summary:
                        values = pd.to_numeric(summary[col], errors="coerce").dropna().unique()
                        if len(values) == 1:
                            diag[col] = float(values[0])
                        elif len(values) > 1:
                            issues.append(f"Inconsistent {col} across parameter rows")
                diag["diagnostics_complete"] = all(np.isfinite(r[k]) for r in local for k in ("r_hat", "ess_bulk", "ess_tail")) and np.isfinite(diag["n_divergences"])
                failed = (diag["max_r_hat"] > MAX_RHAT or diag["min_ess_bulk"] < MIN_ESS_BULK or
                          diag["min_ess_tail"] < MIN_ESS_TAIL or diag["n_divergences"] > 0)
                diag["flagged_convergence"] = True if failed else (False if diag["diagnostics_complete"] else np.nan)
            except Exception as exc:
                issues.append(f"Parameters: {type(exc).__name__}: {exc}")
            if not args.summary_only:
                try:
                    data = pd.read_csv(csv_file)
                except Exception as exc:
                    data = None
                    issues.append(f"CSV: {exc}")
                if model_type in BAYES_MODELS and data is not None:
                    try:
                        b, t, e = analyze_beliefs(data, belief_file, model_type, parameter_set, args.event_window)
                        beliefs.extend(b)
                        trial_beliefs.extend(t)
                        event_beliefs.extend(e)
                        diag["beliefs_analyzed"] = True
                    except Exception as exc:
                        issues.append(f"Beliefs: {type(exc).__name__}: {exc}")
                if args.with_choice_metrics and data is not None:
                    try:
                        diag.update(choice_metrics(posterior_file, data))
                    except Exception as exc:
                        issues.append(f"Choices: {type(exc).__name__}: {exc}")
            diag["problem"] = "; ".join(issues)
            diagnostics.append(diag)
            if issues:
                print(f"  {stem}: {diag['problem']}", flush=True)
    tables = {"fit_diagnostics": pd.DataFrame(diagnostics), "parameter_estimates": pd.DataFrame(estimates),
              "belief_recovery_by_dataset": pd.DataFrame(beliefs), "belief_recovery_by_trial": pd.DataFrame(trial_beliefs),
              "belief_recovery_by_event": pd.DataFrame(event_beliefs)}
    tables["parameter_recovery_summary"] = summarize_parameters(tables["parameter_estimates"])
    # Keep failed/unknown fits in primary recovery results. Supply a clearly separate diagnostic subset.
    if estimates:
        good = tables["fit_diagnostics"].query("diagnostics_complete == True and flagged_convergence == False")
        passed = tables["parameter_estimates"].merge(good[["model_type", "parameter_set"]], on=["model_type", "parameter_set"])
        tables["parameter_recovery_summary_diagnostics_passed"] = summarize_parameters(passed)
    if beliefs:
        b = tables["belief_recovery_by_dataset"]
        keys = ["model_type", "stage", "target", "state_index"]
        cols = ["pearson_r", "spearman_r", "mae", "rmse", "mean_total_variation", "mean_js_nats", "change_mae", "change_pearson_r"]
        grouped = b.groupby(keys, dropna=False, sort=False)
        tables["belief_recovery_summary"] = grouped[cols].mean().join(grouped.size().rename("n_datasets")).reset_index()
        # Mean r is descriptive, not an inferential pooled correlation; per-dataset r remains available.
    if event_beliefs:
        e = tables["belief_recovery_by_event"]
        keys = ["model_type", "parameter_set", "stage", "event_type", "position"]
        tables["belief_change_by_dataset"] = e.groupby(keys, sort=False).agg(
            mean_total_variation=("total_variation", "mean"), mean_js_nats=("js_nats", "mean"),
            n_events=("event_row_index", "nunique")).reset_index()
        # Equal weight per dataset, rather than weighting datasets with more events more heavily.
        g = tables["belief_change_by_dataset"].groupby(["model_type", "stage", "event_type", "position"], sort=False)
        tables["belief_change_summary"] = g.agg(mean_total_variation=("mean_total_variation", "mean"),
            mean_js_nats=("mean_js_nats", "mean"), n_datasets=("parameter_set", "count")).reset_index()
    if diagnostics:
        g = tables["fit_diagnostics"].groupby("model_type", sort=False)
        tables["model_fitting_summary"] = g.agg(n_simulated_datasets=("parameter_set", "count"),
            n_parameters_analyzed=("parameters_analyzed", "sum"), n_beliefs_analyzed=("beliefs_analyzed", "sum"),
            n_complete_diagnostics=("diagnostics_complete", "sum"),
            fraction_flagged_among_assessed=("flagged_convergence", "mean")).reset_index()
    for name, table in tables.items():
        filename = os.path.join(output_dir, name+".csv")
        table.to_csv(filename, index=False)
        print(f"Saved {filename} ({len(table)} rows)", flush=True)
    if estimates and not args.no_plots:
        save_plots(tables["parameter_estimates"], output_dir)
        print("Saved one PNG and PDF per model/parameter under analysis/plots/<model>/")
    print("Summary CSV precision is retained as saved; rounded summaries limit error/coverage precision.")
    print("Recovery summaries include all available fits; inspect diagnostics and the separate passed subset.")
    print("Event position 0 = first trial under a changed environment; block boundaries are excluded.")
    print("Peak changes are defined by a change in peak_idx; map changes by ring_label.")
    print("Choice scores, if requested, are in-sample; belief NPZs contain plug-in trajectories only.")


if __name__ == "__main__":
    main()
