import os
import glob
import json
import numpy as np
import pandas as pd
import arviz as az
import model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


path = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model"
data_path = os.path.join(path, "simulation_results_100_fixedbeta_fixedhazardrate")
fit_path = os.path.join(path, "fitting_results", "parameter_recovery_100_fixedbeta_fixedhazardrate")
output_dir = os.path.join(fit_path, "analysis")

models = ["flat_NoDecay", "flat_Decay", "hybrid_FreeWeights_NoDecay",
          "hybrid_FreeWeights_Decay", "hybrid_Pearce_Hall_NoDecay",
          "hybrid_Pearce_Hall_Decay", "full_Bayesian"]
map_labels = ["map_one", "map_two"]
hdi_probability = 0.95
event_window = 5
max_rhat = 1.01
min_ess_per_chain = 100
min_bfmi = 0.3
fixed_hazards = {"h_map": 4.5 / 83, "h_peak": 4.5 / (83 - 4.5)}

# compute correlations
def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(pearsonr(x, y).statistic)

# get mean of finite values, return NaN if no finite values, used in computing HDI ranges
def mean_finite(x):
    x = np.asarray(x, dtype = float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else np.nan

# check model fitting performance
def fit_diagnostics(idata, summary, summary_file):
    stats = idata.sample_stats
    chains = idata.posterior.sizes["chain"]
    ess_limit = min_ess_per_chain * chains
    bfmi = np.asarray(az.bfmi(idata)) if "energy" in stats else np.array([np.nan])
    divergences = float(stats.diverging.sum()) if "diverging" in stats else np.nan
    values = summary[["r_hat", "ess_bulk", "ess_tail"]].to_numpy()
    failures = {
        "r_hat": (summary.r_hat >= max_rhat).any(),
        "ess_bulk": (summary.ess_bulk < ess_limit).any(),
        "ess_tail": (summary.ess_tail < ess_limit).any(),
        "divergences": divergences > 0,
        "bfmi": (bfmi < min_bfmi).any()
    }
    complete = np.isfinite(values).all() and np.isfinite(bfmi).all() and np.isfinite(divergences)
    status = "failed" if any(failures.values()) else ("passed" if complete else "not_assessed")
    result = dict(n_chains = chains, ess_threshold = ess_limit,
                  max_r_hat = summary.r_hat.max(), min_ess_bulk = summary.ess_bulk.min(),
                  min_ess_tail = summary.ess_tail.min(), n_divergences = divergences,
                  divergence_rate = divergences / (chains * idata.posterior.sizes["draw"]),
                  min_bfmi = np.min(bfmi), diagnostics_complete = complete,
                  diagnostic_status = status,
                  failed_checks = ";".join(name for name, failed in failures.items() if failed),
                  n_max_treedepth = np.nan, max_treedepth_rate = np.nan)
    for chain, value in zip(idata.posterior.chain.values, bfmi):
        result[f"bfmi_chain_{chain}"] = value
    if "reached_max_treedepth" in stats:
        result["n_max_treedepth"] = int(stats.reached_max_treedepth.sum())
        result["max_treedepth_rate"] = float(stats.reached_max_treedepth.mean())
    elif os.path.isfile(summary_file):
        saved_summary = pd.read_csv(summary_file, index_col = 0)
        for name in ("n_max_treedepth", "max_treedepth_rate"):
            if name in saved_summary:
                result[name] = saved_summary[name].iloc[0]
    return result

# summarize parameter recovery results
def summarize_parameters(estimates):
    rows = []
    for (model_type, parameter), g in estimates.groupby(["model_type", "parameter"], sort = False):
        error = g.estimated_mean.to_numpy() - g.true_value.to_numpy()
        rows.append(dict(model_type = model_type, parameter = parameter, n_datasets = len(g),
                         pearson_r = correlation(g.true_value.to_numpy(), g.estimated_mean.to_numpy()),
                         mae = np.abs(error).mean(), bias = error.mean(),
                         hdi_coverage = g.hdi_contains_true.mean(), mean_hdi_width = g.hdi_width.mean()))
    return pd.DataFrame(rows, columns = ["model_type", "parameter", "n_datasets", "pearson_r",
                                        "mae", "bias", "hdi_coverage", "mean_hdi_width"])

# plot parameter recovery results
def plot_parameters(estimates):
    for (model_type, parameter), g in estimates.groupby(["model_type", "parameter"], sort = False):
        folder = os.path.join(output_dir, "plots", model_type)
        os.makedirs(folder, exist_ok = True)
        fig, ax = plt.subplots(figsize = (6, 6))
        ax.vlines(g.true_value, g.hdi_lower, g.hdi_upper, color = "#619CFF", alpha = 0.25)
        ax.scatter(g.true_value, g.estimated_mean, color = "#619CFF", s = 20)
        lo = min(g.true_value.min(), g.hdi_lower.min(), g.estimated_mean.min())
        hi = max(g.true_value.max(), g.hdi_upper.max(), g.estimated_mean.max())
        pad = max((hi - lo) * 0.05, 0.001)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color = "grey")
        ax.set(xlabel = "True parameter", ylabel = "Fitted posterior mean",
               title = f"{model_type}: {parameter}", xlim = (lo - pad, hi + pad),
               ylim = (lo - pad, hi + pad))
        ax.set_aspect("equal", adjustable = "box")
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left = 0.16, bottom = 0.14, top = 0.9)
        fig.savefig(os.path.join(folder, f"{parameter}.png"), dpi = 180)
        plt.close(fig)

# compute trial-wise Bayesian beliefs from posterior samples
def belief_draws(data, metadata, idata, belief_file):
    posterior = idata.posterior
    if "joint_prior" in posterior and "joint_posterior" in posterior:
        result = {}
        for stage in ("joint_prior", "joint_posterior"):
            variable = posterior[stage]
            dimensions = [d for d in variable.dims if d not in ("chain", "draw")]
            array = variable.transpose("chain", "draw", *dimensions).values
            result[stage] = array.reshape((-1,) + array.shape[2:])
        return result

    saved = {}
    if os.path.isfile(belief_file):
        with np.load(belief_file, allow_pickle = False) as file:
            saved = {name: file[name] for name in file.files}
    free_hazards = any(name in posterior for name in fixed_hazards)
    if "joint_prior" in saved and (not free_hazards or saved["joint_prior"].ndim > 3):
        return {stage: saved[stage].reshape((-1,) + saved[stage].shape[-3:])
                for stage in ("joint_prior", "joint_posterior")}
 
    n_peaks = len(json.loads(data.joint_prior.iloc[0])[0])
    n_monsters = len(metadata["map_info"][map_labels[0]])
    angles = np.zeros((2, n_monsters))
    for i, name in enumerate(map_labels):
        angles[i, metadata["map_info"][name]] = np.arange(n_monsters) * 2 * np.pi / n_monsters
    reward_matrix, _ = model.map_reward_matrix(angles, n_peaks = n_peaks, kappa = 1.0)
    transition = model.peak_change_matrix(n_peaks = n_peaks)
    n_samples = posterior.sizes["chain"] * posterior.sizes["draw"] if free_hazards else 1
    hazards = {}
    for name, default in fixed_hazards.items():
        if name in posterior:
            hazards[name] = posterior[name].transpose("chain", "draw").values.reshape(-1, 1, 1)
        else:
            value = posterior.attrs.get(name, saved.get(name, default))
            hazards[name] = np.full((n_samples, 1, 1), float(value))
    h_map, h_peak = hazards["h_map"], hazards["h_peak"]
    priors = np.empty((n_samples, len(data), 2, n_peaks))
    posteriors = np.empty_like(priors)
    previous_block = None
    for t, row in enumerate(data.itertuples(index = False)):
        if t == 0 or row.blockID != previous_block:
            prior = np.full((n_samples, 2, n_peaks), 1.0 / (2 * n_peaks))
        priors[:, t] = prior
        chosen = int(row.left_stim if row.choice == 0 else row.right_stim)
        reward = row.left_reward if row.choice == 0 else row.right_reward
        likelihood = model.full_Bayes_feedback_likelihood(reward, chosen, reward_matrix, obs_sd = 5.0)
        post = prior * likelihood
        post = post / post.sum(axis = (1, 2), keepdims = True)
        posteriors[:, t] = post
        changed = post @ transition
        prior = (1 - h_map) * ((1 - h_peak) * post + h_peak * changed) + h_map * changed[:, ::-1, :]
        previous_block = row.blockID
    return {"joint_prior": priors, "joint_posterior": posteriors}

# analyze trial-wise beliefs 
def analyze_beliefs(data, draws, base, trial_file, write_header):
    rows = []
    same_block = data.blockID.eq(data.blockID.shift()).to_numpy()
    actual_map = data.ring_label.map({name: i for i, name in enumerate(map_labels)}).to_numpy()
    actual_peak = data.peak_idx.to_numpy(dtype = int)
    context_change = data.ring_label.ne(data.ring_label.shift()).to_numpy() & same_block
    peak_only_change = data.peak_idx.ne(data.peak_idx.shift()).to_numpy() & same_block & ~context_change
    block_trial = data.groupby("blockID", sort = False).cumcount().to_numpy() + 1
    block_id = data.blockID.to_numpy()
    event_indices = {"context_change": np.flatnonzero(context_change),
                     "peak_only_change": np.flatnonzero(peak_only_change)}
    positions = np.arange(-event_window, event_window + 1)
    windows = {}
    for event, indices in event_indices.items():
        window = indices[:, None] + positions
        valid = (window >= 0) & (window < len(data))
        clipped = np.clip(window, 0, len(data) - 1)
        valid &= block_id[clipped] == block_id[indices, None]
        windows[event] = (clipped, valid)

    for stage, joint_draws in draws.items():
        true_joint = np.stack([json.loads(value) for value in data[stage]])
        if joint_draws.shape[1:] != true_joint.shape:
            raise ValueError("Belief arrays must have matching trial, map and peak axes.")
        for target, axis in (("map", 3), ("peak", 2)):
            samples = joint_draws.sum(axis = axis)
            true = true_joint.sum(axis = axis - 1)
            fitted = samples.mean(axis = 0)
            posterior_interval = len(samples) > 1 and np.any(np.ptp(samples, axis = 0) > 1e-12)
            if posterior_interval:
                interval = az.hdi(samples[None, ...], hdi_prob = hdi_probability)
                lower, upper = interval[..., 0], interval[..., 1]
            else:
                lower, upper = fitted.copy(), fitted.copy()
            covered = ((lower <= true) & (true <= upper)).astype(float)
            if not posterior_interval:
                covered[:] = np.nan
            width = upper - lower
            tv = 0.5 * np.abs(true - fitted).sum(axis = 1)
            interval_kind = "posterior" if posterior_interval else "fixed"
            states = [0] if target == "map" else range(true.shape[1])

            frames = []
            # distribution-level TV is only stored once on state 0, rather than repeated for every state.
            for state in states:
                frame = pd.DataFrame(dict(row_index = np.arange(len(data)), blockID = block_id,
                                          trialID = data.trialID.to_numpy(), block_trial = block_trial,
                                          context_change = context_change, peak_only_change = peak_only_change,
                                          actual_map = actual_map, actual_peak = actual_peak,
                                          stage = stage, target = target, state_index = state,
                                          true_probability = true[:, state], estimated_probability = fitted[:, state],
                                          hdi_lower = lower[:, state], hdi_upper = upper[:, state],
                                          hdi_contains_true = covered[:, state], hdi_width = width[:, state],
                                          hdi_degenerate = width[:, state] <= 1e-12,
                                          interval_kind = interval_kind,
                                          total_variation = tv if state == 0 else np.nan))
                frames.append(frame.assign(**base))
            pd.concat(frames, ignore_index = True).to_csv(
                trial_file, index = False, mode = "w" if write_header else "a", header = write_header)
            write_header = False

            def add_summary(indices, scope, position = np.nan, include_correlations = False,
                            event_count = np.nan, actual_probability = np.nan, most_likely_probability = np.nan):
                if len(indices) == 0:
                    return
                record = dict(base, level = "dataset", subset = "all", stage = stage, target = target,
                              state_index = 0 if target == "map" else -1, scope = scope, position = position,
                              n_datasets = 1, n_trials = len(indices), n_events = event_count,
                              pearson_r = correlation(true[indices, 0], fitted[indices, 0])
                              if target == "map" and include_correlations else np.nan,
                              mean_total_variation = tv[indices].mean(),
                              hdi_coverage = mean_finite(covered[indices]), mean_hdi_width = width[indices].mean(),
                              fraction_degenerate = (width[indices] <= 1e-12).mean(), interval_kind = interval_kind,
                              actual_probability = actual_probability, most_likely_probability = most_likely_probability)
                rows.append(record)
                if target == "peak" and include_correlations:
                    for state in states:
                        rows.append(dict(record, state_index = state,
                                         pearson_r = correlation(true[indices, state], fitted[indices, state]),
                                         mean_total_variation = np.nan,
                                         hdi_coverage = mean_finite(covered[indices, state]),
                                         mean_hdi_width = width[indices, state].mean(),
                                         fraction_degenerate = (width[indices, state] <= 1e-12).mean()))

            add_summary(np.arange(len(data)), "all_trials", include_correlations = True)
            for event, (window, valid) in windows.items():
                add_summary(np.unique(window[valid]), event + "_window", include_correlations = True)
                anchors = (actual_map if target == "map" else actual_peak)[event_indices[event]]
                for j, position in enumerate(positions):
                    indices = window[valid[:, j], j]
                    anchor = anchors[valid[:, j]]
                    if len(indices):
                        add_summary(indices, event, position, event_count = len(indices),
                                    actual_probability = fitted[indices, anchor].mean(),
                                    most_likely_probability = fitted[indices].max(axis = 1).mean())
            for position in np.unique(block_trial):
                add_summary(np.flatnonzero(block_trial == position), "block_initialization", position)
    return rows, write_header

# summarize beliefs 
def summarize_beliefs(dataset_results):
    keys = ["model_type", "stage", "target", "state_index", "scope", "position"]
    metrics = ["pearson_r", "mean_total_variation", "hdi_coverage", "mean_hdi_width",
               "fraction_degenerate", "actual_probability", "most_likely_probability"]
    summaries = []
    for subset, data in (("all", dataset_results),
                         ("diagnostics_passed", dataset_results[dataset_results.diagnostic_status == "passed"])):
        if data.empty:
            continue
        groups = data.groupby(keys, dropna = False, sort = False)
        result = groups[metrics].mean()
        result["n_datasets"] = groups.parameter_set.nunique()
        result["n_trials"] = groups.n_trials.sum()
        result["n_events"] = groups.n_events.sum(min_count = 1)
        result["interval_kind"] = groups.interval_kind.agg(lambda x: x.iloc[0] if x.nunique() == 1 else "mixed")
        summaries.append(result.reset_index().assign(level = "model", subset = subset))
    # Model-level Pearson r is a descriptive mean of dataset correlations.
    return pd.concat([dataset_results] + summaries, ignore_index = True)

# plot trial-wise belief updates around change points
def plot_belief_updates(dataset_results):
    positions = np.arange(-event_window, event_window + 1)
    lines = [("map", "actual_probability", "Post-change map", "#F8766D"),
             ("peak", "actual_probability", "Post-change peak", "#00BA38"),
             ("map", "most_likely_probability", "Most likely map", "#619CFF"),
             ("peak", "most_likely_probability", "Most likely peak", "#C77CFF")]
    for model_type, model_data in dataset_results.groupby("model_type", sort = False):
        folder = os.path.join(output_dir, "plots", model_type)
        os.makedirs(folder, exist_ok = True)
        for event, title in (("context_change", "Context change"), ("peak_only_change", "Peak-only change")):
            event_data = model_data[model_data.scope == event]
            if event_data.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
            for ax, stage, stage_title in zip(axes, ("joint_prior", "joint_posterior"),
                                            ("Before feedback", "After feedback")):
                for target, column, label, color in lines:
                    g = event_data[(event_data.stage == stage) & (event_data.target == target)]
                    if g.empty:
                        continue
                    values = g.pivot(index = "parameter_set", columns = "position", values = column).reindex(columns = positions)
                    means = values.mean(axis = 0).to_numpy()
                    ax.plot(positions, means, "o-", color = color, label = label, markersize = 4)
                    # Pointwise 95% bootstrap CIs across dataset means, not belief HDIs.
                    lower, upper = np.full(len(positions), np.nan), np.full(len(positions), np.nan)
                    rng = np.random.default_rng(1)
                    for j in range(len(positions)):
                        x = values.iloc[:, j].dropna().to_numpy()
                        if len(x) > 1:
                            bootstrap = x[rng.integers(0, len(x), size = (2000, len(x)))].mean(axis = 1)
                            lower[j], upper[j] = np.quantile(bootstrap, [0.025, 0.975])
                    ax.fill_between(positions, lower, upper, color = color, alpha = 0.18)
                ax.axvline(0, color = "grey", linestyle = ":")
                ax.set(xlabel = "Trial position relative to change", title = stage_title,
                       xticks = positions, ylim = (0, 1), xlim = (-event_window, event_window))
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].set_ylabel("Belief probability")
            fig.suptitle(f"{model_type}: {title}")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc = "lower center", ncol = 4, frameon = False)
            fig.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.2, top = 0.82, wspace = 0.12)
            fig.savefig(os.path.join(folder, f"{event}_beliefs.png"), dpi = 180)
            plt.close(fig)

# main analysis function
def analyze_parameter_recovery():
    os.makedirs(output_dir, exist_ok = True)
    diagnostics, estimates, belief_results = [], [], []
    trial_file = os.path.join(output_dir, "belief_recovery_by_trial.csv")
    write_header = True
    for model_type in models:
        files = sorted(glob.glob(os.path.join(data_path, model_type, "double_map",
                                             f"{model_type}_double_parameter_set_*.csv")))
        for csv_file in files:
            stem = os.path.splitext(os.path.basename(csv_file))[0]
            parameter_set = int(stem.rsplit("_", 1)[1])
            base = dict(model_type = model_type, parameter_set = parameter_set)
            folder = os.path.join(fit_path, model_type)
            posterior_file = os.path.join(folder, stem + "_posterior.nc")
            if not os.path.isfile(posterior_file):
                diagnostics.append(dict(base, diagnostic_status = "not_assessed", failed_checks = "missing_posterior"))
                continue
            data = pd.read_csv(csv_file)
            metadata_file = os.path.join(data_path, model_type, f"{model_type}_parameter_set_{parameter_set:06d}_metadata.json")
            with open(metadata_file) as file:
                metadata = json.load(file)
            idata = az.from_netcdf(posterior_file)
            parameters = [name for name, value in idata.posterior.data_vars.items()
                          if set(value.dims) == {"chain", "draw"}]
            summary = az.summary(idata, var_names = parameters, hdi_prob = hdi_probability, round_to = "none")
            diag = dict(base, **fit_diagnostics(idata, summary, os.path.join(folder, stem + "_summary.csv")))
            diagnostics.append(diag)
            base["diagnostic_status"] = diag["diagnostic_status"]
            hdi_columns = [name for name in summary if name.startswith("hdi_")]
            for parameter in parameters:
                true = metadata["parameters"][parameter] if parameter in metadata["parameters"] else data[parameter].iloc[0]
                row = summary.loc[parameter]
                lower, upper = row[hdi_columns].to_numpy()
                estimates.append(dict(base, parameter = parameter, true_value = true, estimated_mean = row["mean"],
                                      error = row["mean"] - true, absolute_error = abs(row["mean"] - true),
                                      hdi_lower = lower, hdi_upper = upper, hdi_width = upper - lower,
                                      hdi_contains_true = float(lower <= true <= upper)))
            if model_type.startswith("hybrid_") or model_type == "full_Bayesian":
                belief_file = os.path.join(folder, "beliefs", f"parameter_set_{parameter_set:06d}_beliefs.npz")
                draws = belief_draws(data, metadata, idata, belief_file)
                rows, write_header = analyze_beliefs(data, draws, base, trial_file, write_header)
                belief_results.extend(rows)
                del draws
            idata.close()

    diagnostics = pd.DataFrame(diagnostics)
    estimates = pd.DataFrame(estimates)
    diagnostics.to_csv(os.path.join(output_dir, "fit_diagnostics.csv"), index = False)
    fitting_summary = diagnostics.groupby("model_type", sort = False).agg(
        n_datasets = ("parameter_set", "size"),
        n_passed = ("diagnostic_status", lambda x: (x == "passed").sum()),
        n_failed = ("diagnostic_status", lambda x: (x == "failed").sum()),
        n_not_assessed = ("diagnostic_status", lambda x: (x == "not_assessed").sum())).reset_index()
    fitting_summary["fraction_passed"] = fitting_summary.n_passed / fitting_summary.n_datasets
    fitting_summary.to_csv(os.path.join(output_dir, "model_fitting_summary.csv"), index = False)
    if not estimates.empty:
        estimates.to_csv(os.path.join(output_dir, "parameter_estimates.csv"), index = False)
        summarize_parameters(estimates).to_csv(os.path.join(output_dir, "parameter_recovery_summary.csv"), index = False)
        summarize_parameters(estimates[estimates.diagnostic_status == "passed"]).to_csv(
            os.path.join(output_dir, "parameter_recovery_summary_diagnostics_passed.csv"), index = False)
        plot_parameters(estimates)
    if belief_results:
        belief_results = pd.DataFrame(belief_results)
        summarize_beliefs(belief_results).to_csv(os.path.join(output_dir, "belief_recovery_summary.csv"), index = False)
        plot_belief_updates(belief_results)

if __name__ == "__main__":
    analyze_parameter_recovery()
