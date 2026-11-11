#!/usr/bin/env python3
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# set path
path = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model"
data_path = os.path.join(path, "simulation_results")
fit_path = os.path.join(path, "fitting_results", "parameter_recovery")

# models
model_parameters = {
    "flat_NoDecay": ["alpha", "beta"],
    "flat_Decay": ["beta", "K"],
    "hybrid_FreeWeights_NoDecay": ["alpha", "beta", "h_map", "h_peak", "w_mf", "w_map"],
    "hybrid_FreeWeights_Decay": ["beta", "h_map", "h_peak", "w_mf", "w_map", "K"],
    "hybrid_Pearce_Hall_NoDecay": ["alpha", "beta", "h_map", "h_peak", "eta"],
    "hybrid_Pearce_Hall_Decay": ["beta", "h_map", "h_peak", "eta", "K"],
    "full_Bayesian": ["beta", "h_map", "h_peak"],
}

Bayes_models = {name for name in model_parameters if name.startswith("hybrid_") or name == "full_Bayesian"}

# diagnostic thresholds
max_Rhat = 1.01
min_ESS_bulk = 400
min_ESS_tail = 400

# turn values into floats
def numeric(row, name):
    value = row.get(name, np.nan)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")

# get the HDI columns from the summary DataFrame
def hdi_columns(summary):
    cols = [name for name in summary.columns if name.startswith("hdi_") and name.endswith("%")]
    if len(cols) != 2:
        return None, None
    cols.sort(key=lambda name: float(name[4:-1]))
    return cols[0], cols[1]

# run correations between true and fitting values
def safe_correlation(x, y, method):
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    return float(pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic)


def choice_metrics(idata, data):
    """In-sample, one-step-ahead choice metrics conditioned on observed history.

    The saved 'choice' log-likelihood is log P(observed choice | parameters, history).
    It can be inverted to P(left) using the observed 0=left/1=right code.
    """
    if "log_likelihood" in idata.groups() and "choice" in idata.log_likelihood:
        logp = np.asarray(idata.log_likelihood["choice"].values, dtype=float)
    else:
        logp = np.asarray(idata.posterior["trial_loglik"].values, dtype=float)

    # The first two axes correspond to chain and posterior draw.
    if logp.ndim != 3:
        raise ValueError(f"Expected (chain, draw, trial) log-likelihood; received {logp.shape}")
    choice = data["choice"].to_numpy(dtype=int)
    if logp.shape[-1] != len(choice):
        raise ValueError(f"Log-likelihood has {logp.shape[-1]} trials, CSV has {len(choice)}")

    prob_chosen = np.exp(logp).mean(axis=(0, 1))
    prob_chosen = np.clip(prob_chosen, 1e-300, 1.0)
    actual_left = choice == 0
    prob_left = np.where(actual_left, prob_chosen, 1.0 - prob_chosen)
    predicted_left = prob_left >= 0.5

    return {
        "n_trials": len(choice),
        "choice_accuracy": float(np.mean(predicted_left == actual_left)),
        "choice_log_loss": float(-np.mean(np.log(prob_chosen))),
        "choice_brier_score": float(np.mean((prob_left - actual_left.astype(float)) ** 2)),
    }


def belief_metrics(csv_data, belief_path):
    """Compare the post-feedback joint posteriors (not saved next-trial priors)."""
    true_post = np.stack([
        np.asarray(json.loads(value), dtype=float)
        for value in csv_data["joint_posterior"]
    ])
    with np.load(belief_path, allow_pickle=False) as stored:
        recovered_post = np.asarray(stored["joint_posterior"], dtype=float)

    if true_post.shape != recovered_post.shape:
        raise ValueError(f"Belief shapes disagree: simulated {true_post.shape}, recovered {recovered_post.shape}")

    total_variation = 0.5 * np.abs(true_post - recovered_post).sum(axis=(1, 2))
    midpoint = 0.5 * (true_post + recovered_post)
    eps = 1e-300
    kl_true = np.sum(true_post * np.log(np.clip(true_post, eps, None) / np.clip(midpoint, eps, None)), axis=(1, 2))
    kl_fit = np.sum(recovered_post * np.log(np.clip(recovered_post, eps, None) / np.clip(midpoint, eps, None)), axis=(1, 2))
    js_divergence = 0.5 * (kl_true + kl_fit)

    return {
        "belief_mean_total_variation": float(np.mean(total_variation)),
        "belief_mean_js_nats": float(np.mean(js_divergence)),
    }


def analyze_file(model_type, csv_path, summary_path, posterior_path, belief_path, metadata_path, summary_only):
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    parameter_set = int(filename.rsplit("_", 1)[-1])

    record = {
        "model_type": model_type,
        "parameter_set": parameter_set,
        "max_r_hat": np.nan,
        "min_ess_bulk": np.nan,
        "min_ess_tail": np.nan,
        "n_divergences": np.nan,
        "flagged_convergence": np.nan,
        "n_trials": np.nan,
        "choice_accuracy": np.nan,
        "choice_log_loss": np.nan,
        "choice_brier_score": np.nan,
        "belief_mean_total_variation": np.nan,
        "belief_mean_js_nats": np.nan,
        "problem": "",
    }

    estimates = []
    idata = None
    issues = []
    try:
        saved_summary = pd.read_csv(summary_path, index_col=0)
        with open(metadata_path, encoding="utf-8") as f:
            true_parameters = json.load(f)["parameters"]

        if not summary_only and has_posterior:
            import arviz as az
            idata = az.from_netcdf(posterior_path)

        low_col, high_col = hdi_columns(saved_summary)
        for parameter in MODEL_PARAMETERS[model_type]:
            saved_row = saved_summary.loc[parameter]
            true_value = float(true_parameters[parameter])
            estimated_mean = numeric(saved_row, "mean")
            estimated_sd = numeric(saved_row, "sd")
            if idata is not None and parameter in idata.posterior:
                samples = np.asarray(idata.posterior[parameter].values, dtype=float)
                estimated_mean = float(samples.mean())  # full precision, not rounded CSV summary
                estimated_sd = float(samples.std(ddof=1))

            lower = numeric(saved_row, low_col) if low_col else np.nan
            upper = numeric(saved_row, high_col) if high_col else np.nan
            error = estimated_mean - true_value
            estimates.append({
                "model_type": model_type,
                "parameter_set": parameter_set,
                "parameter": parameter,
                "true_value": true_value,
                "estimated_mean": estimated_mean,
                "estimated_sd": estimated_sd,
                "error": error,
                "absolute_error": abs(error),
                "hdi_lower": lower,
                "hdi_upper": upper,
                "hdi_contains_true": (lower <= true_value <= upper) if np.isfinite(lower) and np.isfinite(upper) else np.nan,
                "r_hat": numeric(saved_row, "r_hat"),
                "ess_bulk": numeric(saved_row, "ess_bulk"),
                "ess_tail": numeric(saved_row, "ess_tail"),
            })

        rhats = np.asarray([r["r_hat"] for r in estimates])
        bulk_ess = np.asarray([r["ess_bulk"] for r in estimates])
        tail_ess = np.asarray([r["ess_tail"] for r in estimates])
        record["max_r_hat"] = float(np.nanmax(rhats)) if np.any(np.isfinite(rhats)) else np.nan
        record["min_ess_bulk"] = float(np.nanmin(bulk_ess)) if np.any(np.isfinite(bulk_ess)) else np.nan
        record["min_ess_tail"] = float(np.nanmin(tail_ess)) if np.any(np.isfinite(tail_ess)) else np.nan

        if idata is not None:
            if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats:
                record["n_divergences"] = int(np.asarray(idata.sample_stats["diverging"].values).sum())

            data = pd.read_csv(csv_path)
            record.update(choice_metrics(idata, data))
            if model_type in BAYES_MODELS and has_beliefs:
                record.update(belief_metrics(data, belief_path))
        elif not has_posterior:
            issues.append("Missing posterior NetCDF")

        checks = [
            np.isfinite(record["max_r_hat"]) and record["max_r_hat"] > MAX_RHAT,
            np.isfinite(record["min_ess_bulk"]) and record["min_ess_bulk"] < MIN_ESS_BULK,
            np.isfinite(record["n_divergences"]) and record["n_divergences"] > 0,
        ]
        record["flagged_convergence"] = bool(any(checks))
        if model_type in BAYES_MODELS and not has_beliefs:
            issues.append("Missing fitted beliefs NPZ")

    except Exception as exc:
        # Preserve estimates/diagnostics for other files if one output is corrupt.
        issues.append(f"{type(exc).__name__}: {exc}")
    finally:
        if idata is not None and hasattr(idata, "close"):
            idata.close()

    record["problem"] = "; ".join(issues)
    return record, estimates


def summarize_parameters(estimates):
    results = []
    for (model_type, parameter), group in estimates.groupby(["model_type", "parameter"], sort=False):
        true = group["true_value"].to_numpy(dtype=float)
        fitted = group["estimated_mean"].to_numpy(dtype=float)
        err = fitted - true
        coverage = pd.to_numeric(group["hdi_contains_true"], errors="coerce")
        results.append({
            "model_type": model_type,
            "parameter": parameter,
            "n_fits": len(group),
            "n_distinct_true_values": len(np.unique(true)),
            "pearson_r": safe_correlation(true, fitted, "pearson"),
            "spearman_r": safe_correlation(true, fitted, "spearman"),
            "bias": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_absolute_error": float(np.median(np.abs(err))),
            "hdi_coverage": float(coverage.mean()) if coverage.notna().any() else np.nan,
            "fraction_r_hat_above_1_01": float(np.mean(group["r_hat"] > MAX_RHAT)),
            "fraction_ess_bulk_below_400": float(np.mean(group["ess_bulk"] < MIN_ESS_BULK)),
        })
    return pd.DataFrame(results)


def summarize_models(diagnostics):
    rows = []
    for model_type, group in diagnostics.groupby("model_type", sort=False):
        done = group[group["complete"]]
        assessed = group[group["flagged_convergence"].notna()]
        rows.append({
            "model_type": model_type,
            "n_simulated_datasets": len(group),
            "n_with_summary": int(group["has_summary"].sum()),
            "n_with_posterior": int(group["has_posterior"].sum()),
            "n_complete": len(done),
            "n_with_convergence_diagnostics": len(assessed),
            "fraction_flagged_convergence": float(assessed["flagged_convergence"].astype(float).mean()) if len(assessed) else np.nan,
            "mean_choice_accuracy": group["choice_accuracy"].mean(),
            "mean_choice_log_loss": group["choice_log_loss"].mean(),
            "mean_choice_brier_score": group["choice_brier_score"].mean(),
            "mean_belief_total_variation": group["belief_mean_total_variation"].mean(),
        })
    return pd.DataFrame(rows)


def save_plots(estimates, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for model_type, parameters in MODEL_PARAMETERS.items():
        sub = estimates[estimates["model_type"] == model_type]
        if sub.empty:
            continue
        ncols = 2
        nrows = (len(parameters) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
        for ax, parameter in zip(axes.flat, parameters):
            values = sub[sub["parameter"] == parameter]
            true = values["true_value"].to_numpy(dtype=float)
            fitted = values["estimated_mean"].to_numpy(dtype=float)
            ax.scatter(true, fitted, alpha=0.55, s=18)
            lo = float(min(np.min(true), np.min(fitted)))
            hi = float(max(np.max(true), np.max(fitted)))
            margin = max((hi - lo) * 0.06, 0.01)
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], linestyle="--", linewidth=1)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlabel("Generating value")
            ax.set_ylabel("Fitted posterior mean")
            r = safe_correlation(true, fitted, "pearson")
            ax.set_title(f"{parameter}: n={len(values)}, r={r:.2f}, MAE={np.mean(np.abs(fitted - true)):.3f}")
        for ax in list(axes.flat)[len(parameters):]:
            ax.set_visible(False)
        fig.suptitle(f"Parameter recovery: {model_type}", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{model_type}.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=list(MODEL_PARAMETERS),
                        help="Evaluate just this generating model (repeat to select several)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum datasets per generating model (pilot)")
    parser.add_argument("--summary-only", action="store_true", help="Read only summary CSVs; skip NetCDF and belief metrics")
    parser.add_argument("--no-plots", action="store_true", help="Skip recovery scatter plots")
    parser.add_argument("--output-dir", default=os.path.join(FIT_ROOT, "analysis"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    selected_models = args.model if args.model else list(MODEL_PARAMETERS)
    diagnostics = []
    estimates = []

    for model_type in selected_models:
        pattern = os.path.join(DATA_ROOT, model_type, "double_map", f"{model_type}_double_parameter_set_*.csv")
        files = sorted(glob.glob(pattern))
        if args.limit is not None:
            files = files[:args.limit]
        print(f"{model_type}: {len(files)} simulated double-map datasets", flush=True)
        for i, csv_file in enumerate(files, start=1):
            filename = os.path.splitext(os.path.basename(csv_file))[0]
            parameter_set = int(filename.rsplit("_", 1)[-1])
            fit_dir = os.path.join(FIT_ROOT, model_type)
            summary_path = os.path.join(fit_dir, f"{filename}_summary.csv")
            posterior_path = os.path.join(fit_dir, f"{filename}_posterior.nc")
            belief_path = os.path.join(fit_dir, "beliefs", f"parameter_set_{parameter_set:06d}_beliefs.npz")
            metadata_path = os.path.join(DATA_ROOT, model_type, f"{model_type}_parameter_set_{parameter_set:06d}_metadata.json")

            diag, est = analyze_file(model_type, csv_file, summary_path, posterior_path,
                                     belief_path, metadata_path, args.summary_only)
            diagnostics.append(diag)
            estimates.extend(est)
            if diag["problem"]:
                print(f"  {filename}: {diag['problem']}", flush=True)
            if i % 50 == 0:
                print(f"  Checked {i}/{len(files)}", flush=True)

    if not diagnostics:
        print("No matching simulation CSV files found under:", DATA_ROOT)
        return

    diagnostics = pd.DataFrame(diagnostics)
    estimates = pd.DataFrame(estimates)
    recovery_summary = summarize_parameters(estimates) if not estimates.empty else pd.DataFrame()
    model_summary = summarize_models(diagnostics) if not diagnostics.empty else pd.DataFrame()

    for name, frame in [
        ("fit_diagnostics.csv", diagnostics),
        ("parameter_estimates.csv", estimates),
        ("parameter_recovery_summary.csv", recovery_summary),
        ("model_fitting_summary.csv", model_summary),
    ]:
        path = os.path.join(args.output_dir, name)
        frame.to_csv(path, index=False)
        print(f"Saved {path} ({len(frame)} rows)")

    if not args.no_plots and not estimates.empty:
        save_plots(estimates, args.output_dir)
        print(f"Saved scatter plots in {os.path.join(args.output_dir, 'plots')}")
    if not model_summary.empty:
        print("\nFit completion summary:")
        print(model_summary[["model_type", "n_simulated_datasets", "n_complete"]].to_string(index=False))
    if args.summary_only:
        print("\nSummary-only mode: choice fit, divergences, and belief distances were not calculated.")
    print("\nChoice scores are IN-SAMPLE; they are not cross-validated predictive performance or model-selection statistics.")


if __name__ == "__main__":
    main()
