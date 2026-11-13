import os
import glob
import numpy as np
import pandas as pd
from ast import literal_eval
import bambi as bmb
import arviz as az
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

# compute accuracy
def compute_accuracy(data):
    data = data.copy()
    data["best_reward"] = data[["left_reward", "right_reward"]].max(axis = 1)
    data["accuracy"] = (data["reward"] == data["best_reward"]).astype(float)

    return data

# extract trials around changes
def extract_change_windows(data, task_label, window = 5):
    rows = []
    event_id = 0

    # changes are detected separately within each block
    for block_id, block in data.groupby("blockID"):
        block = block.sort_values("trialID").reset_index(drop = True)

        previous_peak = block["peak_idx"].shift(1)
        previous_map = block["ring_label"].shift(1)

        peak_change = ((block["peak_idx"] != previous_peak) & previous_peak.notna())

        if task_label in ["single_map_one", "single_map_two"]:
            events = [(position, "peak_change") for position in np.where(peak_change)[0]]
        else:
            context_change = ((block["ring_label"] != previous_map) & previous_map.notna())
            peak_only_change = peak_change & ~context_change
            events = [(position, "peak_only_change") for position in np.where(peak_only_change)[0]]
            events += [ (position, "context_change") for position in np.where(context_change)[0]]

        for position, change_type in events:
            if position - window < 0:
                continue

            if position + window >= len(block):
                continue

            event_id += 1
            event_window = block.iloc[position - window:position + window + 1].copy()
            event_window["relative_trial"] = np.arange(-window, window + 1)
            event_window["change_type"] = change_type
            event_window["event_id"] = event_id

            rows.append(event_window)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index = True)

# prepare choice data for the linear mixed model, Day 3
def build_choice_predictors_single(simulation_data):
    data = simulation_data.loc[simulation_data['task'].isin(['choice task single ring_one', 'choice task single ring_two'])].copy()
    data["ring_label"] = data["ring_label"].replace({"map_one": "ring_one", "map_two": "ring_two"})

    results = []
    for (model_type, param_set, block), block_data in data.groupby(['model_type', 'parameter_set', 'blockID']):
        rings = {name: literal_eval(part['ring'].iloc[0]) for name, part in block_data.groupby('ring_label')}
        ring_index = {name: {stim: i for i, stim in enumerate(order)} for name, order in rings.items()}

        for ring_label, g in block_data.groupby('ring_label'):
            g = g.sort_values('trialID').reset_index(drop = True).copy()
            g['trialID'] = g['trialID'].astype(int)
            g["choice"] = 1 - g["choice"].astype(int) # left = 0 in the simulation
            indices = ring_index[ring_label]
            opposite_indices = ring_index['ring_two' if ring_label == 'ring_one' else 'ring_one']
            g['dist'] = g['reward_dist'].map(literal_eval)

            # all value differences are based on identities of the two offered stimuli.
            def diff(dist, idx, left, right):
                if not isinstance(dist, list):
                    return np.nan
                return (dist[idx[left]] - dist[idx[right]]) / 100

            g['R1_diff'] = [diff(d, ring_index['ring_one'], l, r) for d, l, r in zip(g['dist'], g['left_stim'], g['right_stim'])]
            g['R2_diff'] = [diff(d, ring_index['ring_two'], l, r) for d, l, r in zip(g['dist'], g['left_stim'], g['right_stim'])]
            g['rel_diff'] = g['R1_diff'] if ring_label == 'ring_one' else g['R2_diff']
            g['irrel_diff'] = g['R2_diff'] if ring_label == 'ring_one' else g['R1_diff']

            # experience-based values: update the chosen stimulus only AFTER the trial.
            q = {stim: 50.0 for stim in range(7)}
            experienced = []
            for l, r, choice, reward in zip(g['left_stim'], g['right_stim'], g['choice'], g['reward']):
                experienced.append((q[l] - q[r]) / 100)
                q[l if choice == 1 else r] = reward
            g['experience_diff'] = experienced

            # segment each separate (participant, blockID, ring_label) block.
            g['peak_change'] = g['reward_dist'].ne(g['reward_dist'].shift())
            g.loc[0, 'peak_change'] = False
            g['mini_block'] = g['peak_change'].cumsum() + 1

            mini_dist = g.groupby('mini_block')['dist'].first()
            mini_start = g.groupby('mini_block')['trialID'].min()
            g['previous_change_trial'] = g['mini_block'].map(mini_start.where(mini_start.index > 1))
            g['next_change_trial'] = g['mini_block'].map(mini_start.shift(-1))
            g['position_prev'] = g['trialID'] - g['previous_change_trial']
            g['position_next'] = g['trialID'] - g['next_change_trial']

            # for an event at the start of the current mini-block: use preceding distribution.
            old_dist = g['mini_block'].map(mini_dist.shift(1))
            g['old_diff_for_previous_change'] = [diff(d, indices, l, r) for d, l, r in zip(old_dist, g['left_stim'], g['right_stim'])]
            g['old_opposite_diff_for_previous_change'] = [diff(d, opposite_indices, l, r) for d, l, r in zip(old_dist, g['left_stim'], g['right_stim'])]

            # for an event at the start of the next mini-block: use forthcoming distribution.
            new_dist = g['mini_block'].map(mini_dist.shift(-1))
            g['new_diff_for_next_change'] = [diff(d, indices, l, r) for d, l, r in zip(new_dist, g['left_stim'], g['right_stim'])]
            g['new_opposite_diff_for_next_change'] = [diff(d, opposite_indices, l, r) for d, l, r in zip(new_dist, g['left_stim'], g['right_stim'])]

            results.append(g.drop(columns = 'dist'))

    trialwise_data = pd.concat(results, ignore_index = True)

    # make window data
    before = trialwise_data.loc[trialwise_data['position_next'].between(-5, -1)].copy()
    before['position'] = before['position_next']
    before['change_trial'] = before['next_change_trial']
    before['preswitch_diff'] = before['rel_diff']
    before['postswitch_diff'] = before['new_diff_for_next_change']

    after = trialwise_data.loc[trialwise_data['position_prev'].between(0, 5)].copy()
    after['position'] = after['position_prev']
    after['change_trial'] = after['previous_change_trial']
    after['preswitch_diff'] = after['old_diff_for_previous_change']
    after['postswitch_diff'] = after['rel_diff']

    peak_change_data = pd.concat([before, after], ignore_index = True).sort_values(['subj_id', 'blockID', 'ring_label', 'change_trial', 'position']).reset_index(drop = True)
    peak_change_data["position"] = peak_change_data["position"].astype(int).astype(str)

    trialwise_data = trialwise_data[['subj_id', 'day', 'task', 'blockID', 'trialID', 'ring_label', 'ring', 'reward_dist',
                                        'left_choice', 'right_choice', 'left_stim', 'right_stim', 'choice', 'experience_diff',
                                        'R1_diff', 'R2_diff', 'rel_diff', 'irrel_diff']]

    peak_change_data = peak_change_data[['subj_id', 'day', 'task', 'blockID', 'trialID', 'ring_label', 'ring', 'reward_dist', 'left_choice', 
                                            'right_choice', 'left_stim', 'right_stim', 'choice', 'position', 'experience_diff', 'preswitch_diff',
                                            'postswitch_diff']]
    
    return trialwise_data, peak_change_data

# prepare choice data for the linear mixed model, Day 4
def build_choice_predictors_double(simulation_data):
    data = simulation_data.loc[simulation_data['task'].eq('choice task two context')].copy()
    data["ring_label"] = data["ring_label"].replace({"map_one": "ring_one", "map_two": "ring_two"})

    trials, context_windows, peak_windows = [], [], []

    for (model_type, param_set, block), block_data in data.groupby(['model_type', 'parameter_set', 'blockID']):
        g = block_data.sort_values('trialID').reset_index(drop = True).copy()
        g['trialID'] = g['trialID'].astype(int)
        g["choice"] = 1 - g["choice"].astype(int) # left = 0 in the simulation

        rings = {label: literal_eval(part['ring'].iloc[0]) for label, part in g.groupby('ring_label')}
        ring_index = {label: {stim: i for i, stim in enumerate(order)} for label, order in rings.items()}
        g['dist'] = g['reward_dist'].map(literal_eval)

        def diff(dist, indices, left, right):
            return (dist[indices[left]] - dist[indices[right]]) / 100

        g['R1_diff'] = [diff(d, ring_index['ring_one'], l, r) for d, l, r in zip(g['dist'], g['left_stim'], g['right_stim'])]
        g['R2_diff'] = [diff(d, ring_index['ring_two'], l, r) for d, l, r in zip(g['dist'], g['left_stim'], g['right_stim'])]
        g['rel_diff'] = np.where(g['ring_label'].eq('ring_one'), g['R1_diff'], g['R2_diff'])
        g['irrel_diff'] = np.where(g['ring_label'].eq('ring_one'), g['R2_diff'], g['R1_diff'])

        # one value per stimulus within the block; feedback comes after the prediction, and all seven values reset at the next block.
        q = {stim: 50.0 for stim in range(7)}
        experienced = []
        for left, right, choice, reward in zip(g['left_stim'], g['right_stim'], g['choice'], g['reward']):
            experienced.append((q[left] - q[right]) / 100)
            q[left if choice == 1 else right] = reward
        g['experience_diff'] = experienced

        # a context change also changes the peak while peak change is peak-only.
        g['context_change'] = g['ring'].ne(g['ring'].shift())
        g.loc[0, 'context_change'] = False
        g['peak_change'] = (g['ring'].eq(g['ring'].shift()) & g['reward_dist'].ne(g['reward_dist'].shift()))
        g['mini_block'] = (g['context_change'] | g['peak_change']).cumsum() + 1

        mini_start = g.groupby('mini_block')['trialID'].first()
        g['previous_change_trial'] = g['mini_block'].map(mini_start.where(mini_start.index > 1))
        g['next_change_trial'] = g['mini_block'].map(mini_start.shift(-1))
        g['position_prev'] = g['trialID'] - g['previous_change_trial']
        g['position_next'] = g['trialID'] - g['next_change_trial']

        # finish every trial's learning update before duplicating observations.
        trials.append(g.drop(columns = 'dist'))

        for i in np.flatnonzero(g['context_change'] | g['peak_change']):
            old, new = g.iloc[i - 1], g.iloc[i]
            change_trial = int(new['trialID'])
            window = g.loc[g['trialID'].between(change_trial - 5, change_trial + 5)].copy()

            old_idx = ring_index[old['ring_label']]
            new_idx = ring_index[new['ring_label'] if new['context_change'] else ('ring_two' if old['ring_label'] == 'ring_one' else 'ring_one')]
            old_dist, new_dist = old['dist'], new['dist']

            window['change_trial'] = change_trial
            window['position'] = window['trialID'] - change_trial
            window['change_type'] = 'context' if new['context_change'] else 'peak'
            window['ring_before_change'] = old['ring_label']
            window['ring_after_change'] = new['ring_label']
            window['reward_dist_before_change'] = old['reward_dist']
            window['reward_dist_after_change'] = new['reward_dist']

            for n, indices, dist in ((1, old_idx, old_dist), (2, old_idx, new_dist), (3, new_idx, old_dist),(4, new_idx, new_dist)):
                window[f'strategy{n}_diff'] = [diff(dist, indices, l, r) for l, r in zip(window['left_stim'], window['right_stim'])]

            window['preswitch_diff'] = window['strategy1_diff']
            window['postswitch_diff'] = (
                window['strategy4_diff'] if new['context_change']
                else window['strategy2_diff']
            )
            
            window = window.drop(columns = 'dist')
            if new['context_change']:
                context_windows.append(window)
            else:
                peak_windows.append(window)

    trialwise_data = pd.concat(trials, ignore_index = True)
    context_change_data = pd.concat(context_windows, ignore_index = True).sort_values(['parameter_set', 'blockID', 'change_trial', 'position']).reset_index(drop = True)
    context_change_data["position"] = context_change_data["position"].astype(int).astype(str)
    peak_change_data = pd.concat(peak_windows, ignore_index = True).sort_values(['parameter_set', 'blockID', 'change_trial', 'position']).reset_index(drop = True)
    peak_change_data["position"] = peak_change_data["position"].astype(int).astype(str)

    trialwise_data = trialwise_data[['model_type', 'parameter_set', 'task', 'blockID', 'trialID', 'ring_label', 'ring', 'reward_dist',
                                     'left_pos', 'right_pos', 'left_stim', 'right_stim', 'choice', 'experience_diff',
                                     'R1_diff', 'R2_diff', 'rel_diff', 'irrel_diff']]

    peak_change_data = peak_change_data[['model_type', 'parameter_set', 'task', 'blockID', 'trialID', 'ring_label', 'ring', 'reward_dist', 'left_pos', 
                                         'right_pos', 'left_stim', 'right_stim', 'choice', 'position', 'experience_diff', 'preswitch_diff',
                                         'postswitch_diff']]

    context_change_data = context_change_data[['model_type', 'parameter_set', 'task', 'blockID', 'trialID', 'ring_label', 'ring', 'reward_dist', 'left_pos', 
                                               'right_pos', 'left_stim', 'right_stim', 'choice', 'position', 'experience_diff', 'preswitch_diff',
                                               'postswitch_diff']]

    return trialwise_data, context_change_data, peak_change_data

# summarize model behavioral data
def summarize_data(file):
    data = pd.read_csv(file)
    data = compute_accuracy(data)
    model_type = data["model_type"].iloc[0]
    parameter_set = data["parameter_set"].iloc[0]
    task_type = data["task_type"].iloc[0]

    # identify the three task types
    if task_type == "single":
        map_label = data["ring_label"].iloc[0]
        if map_label == "map_one":
            task_label = "single_map_one"
        else:
            task_label = "single_map_two"
    else:
        task_label = "double_map"

    overall_result = {
        "model_type": model_type,
        "parameter_set": parameter_set,
        "task": task_label,

        "alpha": data["alpha"].iloc[0],
        "K": data["K"].iloc[0],
        "h_map": data["h_map"].iloc[0],
        "h_peak": data["h_peak"].iloc[0],
        "w_mf": data["w_mf"].iloc[0],
        "w_map": data["w_map"].iloc[0],
        "eta": data["eta"].iloc[0],

        "accuracy": data["accuracy"].mean(),
        "n_trials": data["accuracy"].notna().sum()
    }

    # accuracy around changes
    change_data = extract_change_windows(data = data, task_label = task_label, window = 5)
    parameter_summary = None

    if not change_data.empty:
        parameter_summary = change_data.groupby(["change_type", "relative_trial"], as_index = False).agg(accuracy = ("accuracy", "mean"), n_events = ("event_id", "nunique"))
        parameter_summary["model_type"] = model_type
        parameter_summary["parameter_set"] = parameter_set
        parameter_summary["task"] = task_label

    return overall_result, parameter_summary

# model performance check
# All diagnostics refer to posterior sampling, not predictive model fit.
def check_model(fitted_model):
    summary = az.summary(fitted_model, round_to = "none")

    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "divergences": int(fitted_model.sample_stats["diverging"].sum().item()),
        "min_bfmi": float(np.min(az.bfmi(fitted_model)))
    }

# extract the population-level fixed-effect terms (not the intercept or random effects)
def get_regressors(model):
    component = model.components[model.family.likelihood.parent]
    return list(component.common_terms.keys())

# Bayesian equivalents of the two original coefficient-comparison families
def compare_regressors(model, fitted_model):
    regressors = get_regressors(model)

    beta = {}
    for name in regressors:
        samples = fitted_model.posterior[name]
        beta[name] = samples.values.reshape(-1)

    results = []

    def test(label, samples, family):
        lower, upper = np.quantile(samples, [0.025, 0.975])
        prob_positive = np.mean(samples > 0)

        results.append({
            "family": family,
            "comparison": label,
            "estimate": float(np.mean(samples)),
            "sd": float(np.std(samples, ddof = 1)),
            "ci_low": float(lower),
            "ci_high": float(upper),
            "prob_gt_zero": float(prob_positive),
            "prob_lt_zero": float(np.mean(samples < 0)),
            "supported_95": bool(prob_positive >= 0.95 if family == "above_zero" else lower > 0 or upper < 0)
        })

    # direction-specific posterior probability for each coefficient > 0
    for name in regressors:
        test(name, beta[name], "above_zero")

    # two-sided comparisons of each coefficient pair
    for a, b in combinations(regressors, 2):
        test(f"{a} - {b}", beta[a] - beta[b], "pairwise")

    # Posterior probabilities and credible intervals are not Holm-adjusted p-values.
    return pd.DataFrame(results)

# extract group-level and individual-level beta estimates for plotting
def extract_betas(model, fitted_model, group = "parameter_set"):
    posterior = fitted_model.posterior
    component = model.components[model.family.likelihood.parent]
    terms = (["Intercept"] if component.intercept_term is not None else []) + get_regressors(model)

    # Population-level posterior mean and 95% highest-density credible interval
    group_beta = az.summary(fitted_model, var_names = terms, hdi_prob = 0.95, round_to = "none")
    group_beta.index.name = "term"
    group_beta = group_beta.reset_index()
    group_beta["estimate"] = group_beta["mean"]
    group_beta["conf_low"] = group_beta["hdi_2.5%"]
    group_beta["conf_high"] = group_beta["hdi_97.5%"]

    # The random intercept supplies the full list of simulated individuals.
    intercept_random = f"1|{group}"
    random_samples = posterior[intercept_random]
    dims = [dim for dim in random_samples.dims if dim not in ("chain", "draw")]

    group_dim = dims[0]
    levels = random_samples.coords[group_dim].values
    numeric_levels = pd.to_numeric(levels, errors = "coerce")
    if pd.notna(numeric_levels).all():
        levels = numeric_levels.astype(int) if np.all(numeric_levels == np.floor(numeric_levels)) else numeric_levels
    individual_beta = pd.DataFrame({group: levels})

    for term in terms:
        random_term = f"{'1' if term == 'Intercept' else term}|{group}"
        samples = posterior[term] + posterior[random_term]
        individual_beta[term] = samples.mean(dim = ("chain", "draw")).values

    return individual_beta, group_beta

# extract population-level predictor slopes at each categorical position
# beta_k(position) = beta_k + beta_(k:position), with a zero interaction at the reference position.
def extract_position_effects(model, fitted_model):
    posterior = fitted_model.posterior
    component = model.components[model.family.likelihood.parent]
    regressors = [
        name for name in component.common_terms
        if name != "position"
        and ":" not in name
        and name in model.data.columns
        and pd.api.types.is_numeric_dtype(model.data[name])
    ]
    positions = sorted(model.data["position"].dropna().unique(), key = lambda x: float(x))

    results = []
    for pred in regressors:
        base = posterior[pred]

        interaction_name = f"{pred}:position"
        if interaction_name not in posterior:
            interaction_name = f"position:{pred}"

        interaction = posterior[interaction_name]
        dims = [dim for dim in interaction.dims if dim not in ("chain", "draw")]

        interaction_dim = dims[0]
        levels = {str(value): value for value in interaction.coords[interaction_dim].values}
        reference_levels = set(str(p) for p in positions) - set(levels)

        for p in positions:
            p_label = str(p)
            samples = base if p_label not in levels else base + interaction.sel({interaction_dim: levels[p_label]})
            values = samples.values.reshape(-1)
            lower, upper = np.quantile(values, [0.025, 0.975])

            results.append({
                "position": int(float(p)),
                "predictor": pred,
                "estimate": float(np.mean(values)),
                "sd": float(np.std(values, ddof = 1)),
                "ci_low": float(lower),
                "ci_high": float(upper),
                "prob_gt_zero": float(np.mean(values > 0)),
                "prob_lt_zero": float(np.mean(values < 0))
            })

    return pd.DataFrame(results).sort_values(["position", "predictor"]).reset_index(drop = True)

# run regression models on simulation data
def run_model_regressions(model_type):
    data_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model"
    # model_dir = os.path.join(data_dir, "simulation_results_balanced100_fixedbeta", model_type)
    # output_dir = os.path.join(data_dir, "model_behavior_analysis_balanced100_fixedbeta", "regressions", model_type)
    model_dir = os.path.join(data_dir, "simulation_results_100_fixedbeta_fixedhazardrate", model_type)
    output_dir = os.path.join(data_dir, "model_behavior_analysis_100_fixedbeta_fixedhazardrate", "regressions", model_type)
    os.makedirs(output_dir, exist_ok = True)
    folders = {
        "one": os.path.join(model_dir, "single_map", "map_one"),
        "two": os.path.join(model_dir, "single_map", "map_two"),
        "double": os.path.join(model_dir, "double_map")
    }
    files = {
        name: {
            int(os.path.splitext(os.path.basename(file))[0].rsplit("parameter_set_", 1)[1]): file
            for file in glob.glob(os.path.join(folder, "*.csv"))
        }
        for name, folder in folders.items()
    }

    parameter_sets = set(files["one"])
    data_columns = ["model_type", "parameter_set", "task", "blockID", "trialID",
                     "ring_label", "ring", "reward_dist", "left_pos", "right_pos",
                     "left_stim", "right_stim", "choice", "reward"]

    trial_columns = ["parameter_set", "choice", "experience_diff", "R1_diff", "R2_diff", 
                     "rel_diff", "irrel_diff"]

    window_columns = ["parameter_set", "choice", "position", "experience_diff", "strategy1_diff", 
                      "strategy2_diff", "strategy3_diff", "strategy4_diff"]

    D3_trials, D4_trials = [], []
    D3_peak, D4_peak, D4_context = [], [], []

    for parameter_set in sorted(parameter_sets):

        single_data = pd.concat([
            pd.read_csv(files["one"][parameter_set], usecols = data_columns),
            pd.read_csv(files["two"][parameter_set], usecols = data_columns)
        ], ignore_index = True)

        double_data = pd.read_csv(files["double"][parameter_set], usecols = data_columns)

        d3, p3 = build_choice_predictors_single(single_data)
        d4, c4, p4 = build_choice_predictors_double(double_data)

        D3_trials.append(d3[trial_columns])
        D4_trials.append(d4[trial_columns])
        D3_peak.append(p3[window_columns])
        D4_peak.append(p4[window_columns])
        D4_context.append(c4[window_columns])

    D3_trials = pd.concat(D3_trials, ignore_index = True)
    D4_trials = pd.concat(D4_trials, ignore_index = True)
    D3_peak = pd.concat(D3_peak, ignore_index = True)
    D4_peak = pd.concat(D4_peak, ignore_index = True)
    D4_context = pd.concat(D4_context, ignore_index = True)

    # Uncorrelated varying slopes reproduce the Bambi structure used for your human choice analyses.
    # Compared with the old simulation script, these allow meaningful individual-specific beta values.
    identity_formula = """
        choice ~ experience_diff + R1_diff + R2_diff +
        (1 | parameter_set) +
        (0 + experience_diff | parameter_set) +
        (0 + R1_diff | parameter_set) +
        (0 + R2_diff | parameter_set)
    """

    relevance_formula = """
        choice ~ experience_diff + rel_diff + irrel_diff +
        (1 | parameter_set) +
        (0 + experience_diff | parameter_set) +
        (0 + rel_diff | parameter_set) +
        (0 + irrel_diff | parameter_set)
    """

    change_formula = """
        choice ~ (experience_diff + preswitch_diff + postswitch_diff) * position +
        (1 | subj_id) +
        (0 + experience_diff | subj_id) +
        (0 + preswitch_diff | subj_id) +
        (0 + postswitch_diff | subj_id)
    """

    regressions = {
        "D3_identity": (D3_trials, identity_formula),
        "D3_relevance": (D3_trials, relevance_formula),
        "D4_identity": (D4_trials, identity_formula),
        "D4_relevance": (D4_trials, relevance_formula),
        "D3_peak_change": (D3_peak, change_formula),
        "D4_peak_change": (D4_peak, change_formula),
        "D4_context_change": (D4_context, change_formula)
    }

    results = {
        "model_check": [],
        "statistical_comparison": [],
        "group_beta": [],
        "individual_beta": [],
        "position_beta": []
    }

    n_cores = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))

    for regression, (data, formula) in regressions.items():
        data = data.copy()
        categorical = ["parameter_set"]
        if "position" in data.columns:
            # make positions categorical, with a predictable numerical ordering.
            position_levels = sorted(data["position"].dropna().unique(), key = lambda x: float(x))
            data["position"] = pd.Categorical(data["position"], categories = position_levels, ordered = False)
            categorical.append("position")

        model = bmb.Model(formula, data = data, family = "bernoulli", categorical = categorical)
        fitted_model = model.fit(
            draws = 1000, tune = 1000, chains = 4, cores = n_cores,
            target_accept = 0.95, random_seed = 334424044,
            idata_kwargs = {"log_likelihood": False}
        )

        # MCMC diagnostics
        diagnostics = check_model(fitted_model)
        diagnostics["regression"] = regression
        results["model_check"].append(pd.DataFrame([diagnostics]))

        if regression in ["D3_identity", "D3_relevance", "D4_identity", "D4_relevance"]:

            # statistical comparisons of population-level slopes
            comparison = compare_regressors(model, fitted_model)
            comparison["regression"] = regression
            results["statistical_comparison"].append(comparison)

            # individual-level and individual-level beta estimates
            individual_beta, group_beta = extract_betas(model, fitted_model)
            group_beta["regression"] = regression
            individual_beta["regression"] = regression
            results["group_beta"].append(group_beta)
            results["individual_beta"].append(individual_beta)

        else:

            # position-specific population slopes from posterior draws
            position_beta = extract_position_effects(model, fitted_model)
            position_beta["regression"] = regression
            results["position_beta"].append(position_beta)

        del model, fitted_model

    for name, tables in results.items():
        if tables:
            output = pd.concat(tables, ignore_index = True)
            output.insert(0, "model_type", model_type)
            output.to_csv(os.path.join(output_dir, f"{name}.csv"), index = False)

if __name__ == "__main__":
    # set path
    # data_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/simulation_results_balanced100_fixedbeta"
    # output_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/model_behavior_analysis_balanced100_fixedbeta"
    data_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/simulation_results_100_fixedbeta_fixedhazardrate"
    output_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/model_behavior_analysis_100_fixedbeta_fixedhazardrate"
    os.makedirs(output_dir, exist_ok = True)
    files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive = True)
    overall_results = []
    change_results = []

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        model_types = ["flat_NoDecay", "flat_Decay", "hybrid_FreeWeights_NoDecay",
                       "hybrid_FreeWeights_Decay", "hybrid_Pearce_Hall_NoDecay",
                       "hybrid_Pearce_Hall_Decay", "full_Bayesian"]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_model_regressions(model_types[array_id])
        raise SystemExit(0)

    # parallel processing
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    with ProcessPoolExecutor(max_workers = n_workers, mp_context = get_context("spawn")) as executor:
        for overall_result, parameter_summary in executor.map(summarize_data, files):
            overall_results.append(overall_result)
            if parameter_summary is not None:
                change_results.append(parameter_summary)

    # save parameter-level results
    overall_results = pd.DataFrame(overall_results)
    overall_results = overall_results.sort_values(["model_type", "task", "parameter_set"]).reset_index(drop = True)
    overall_results.to_csv(os.path.join(output_dir, "overall_accuracy.csv"), index = False)

    change_results = pd.concat(change_results, ignore_index = True)
    change_results = change_results[["model_type", "parameter_set", "task", "change_type", "relative_trial", "accuracy", "n_events"]]
    change_results = change_results.sort_values(["model_type", "task", "parameter_set"]).reset_index(drop = True)
    change_results.to_csv(os.path.join(output_dir, "change_accuracy.csv"), index = False)

    # save model-level results
    overall_model_summary = overall_results.groupby(["model_type", "task"], as_index = False).agg(mean_accuracy = ("accuracy", "mean"), sd_accuracy = ("accuracy", "std"), n_parameter_sets = ("accuracy", "count"))
    overall_model_summary["se_accuracy"] = overall_model_summary["sd_accuracy"] / np.sqrt(overall_model_summary["n_parameter_sets"])
    overall_model_summary.to_csv(os.path.join(output_dir, "overall_accuracy_model_summary.csv"), index = False)

    change_model_summary = change_results.groupby(["model_type", "task", "change_type", "relative_trial"], as_index = False).agg(mean_accuracy = ("accuracy", "mean"), sd_accuracy = ("accuracy", "std"), n_parameter_sets = ("accuracy", "count"))
    change_model_summary["se_accuracy"] = change_model_summary["sd_accuracy"] / np.sqrt(change_model_summary["n_parameter_sets"])
    change_model_summary.to_csv(os.path.join(output_dir, "change_accuracy_model_summary.csv"), index = False)
