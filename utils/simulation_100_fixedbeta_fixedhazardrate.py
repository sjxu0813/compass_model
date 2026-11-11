import os
import json
import random
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import truncnorm
import model
import task

# ============================================================
# 1. Model Configurations
# ============================================================

model_configs = {
    # flat_NoDecay: alpha and beta
    "flat_NoDecay": {
        "map_free": "rw",
        "inference": None,
        "arbitration": None,
        "memory_decay": False,
        "parameters": ["alpha", "beta"]
    },

    # flat_Decay: beta, and K
    "flat_Decay": {
        "map_free": "rw",
        "inference": None,
        "arbitration": None,
        "memory_decay": True,
        "parameters": ["beta", "K"]
    },

    # hybrid_NoDecay: alpha, beta, h_map, h_peak, w_mf and w_map
    "hybrid_FreeWeights_NoDecay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "free_weights",
        "memory_decay": False,
        "parameters": ["alpha", "beta", "w_mf", "w_map"]
    },

    # hybrid_Decay: beta, h_map, h_peak, w_mf, w_map and K
    "hybrid_FreeWeights_Decay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "free_weights",
        "memory_decay": True,
        "parameters": ["beta", "w_mf", "w_map", "K"]
    },

    # hybrid_Pearce_Hall_NoDecay: alpha, beta, h_map, h_peak and eta
    "hybrid_Pearce_Hall_NoDecay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "pearce_hall",
        "memory_decay": False,
        "parameters": ["alpha", "beta", "eta"]
    },

    # hybrid_Pearce_Hall_Decay: beta, h_map, h_peak, eta and K
    "hybrid_Pearce_Hall_Decay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "pearce_hall",
        "memory_decay": True,
        "parameters": ["beta", "eta", "K"]
    },

    # full_Bayesian: beta, h_map, h_peak
    "full_Bayesian": {
        "map_free": None,
        "inference": "full_bayes",
        "arbitration": None,
        "memory_decay": False,
        "parameters": ["beta"]
    }
}

def get_model_config(model_type):
    if model_type not in model_configs:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available models: {list(model_configs.keys())}"
        )

    config = model_configs[model_type].copy()

    model.validate_model_config(
        map_free = config["map_free"],
        inference = config["inference"],
        arbitration = config["arbitration"],
        memory_decay = config["memory_decay"]
    )

    return config

# parameter ranges 
parameter_ranges = {
    "alpha": np.arange(0.1, 1.0, 0.1),
    "K": [0.05, 0.20, 0.35, 0.55, 0.80],

    # 0.01 = every 100 trials, 0.03 = every 40 trials, 0.05 = every 20 trials, 0.12 = every 8.4 trials (actual value), 0.2 = every 5 trials
    # "h_map": [0.01, 0.03, 0.05, 0.12, 0.20], 
    # "h_peak": [0.01, 0.03, 0.05, 0.12, 0.20],

    "w_mf": np.arange(0.1, 1, 0.2),
    "w_map": np.arange(0.1, 1, 0.2),
    "eta": [0.05, 0.10, 0.20, 0.40, 0.60]
}

fixed_hazards = {
    "single": {
        "h_map": 0.0,
        "h_peak": 9 / 83
    },
    "double": {
        "h_map": 4.5 / 83,
        "h_peak": 4.5 / (83 - 4.5)
    }
}

def get_parameter_combinations(model_type):
    
    # beta is sampled not given
    parameter_names = [ name for name in model_configs[model_type]["parameters"] if name != "beta"]

    for name in parameter_names:
        if len(parameter_ranges[name]) == 0:
            raise ValueError(f"No parameter range has been defined for '{name}'.")

    combinations = [
        dict(zip(parameter_names, values))
        for values in product( *(parameter_ranges[name] for name in parameter_names))
    ]

    return combinations

# 100 samples per model
def sample_parameter_sets(model_type, seed = 1):
    parameter_combinations = get_parameter_combinations(model_type)
    names = [name for name in model_configs[model_type]["parameters"] if name != "beta"]
    rng = np.random.default_rng(seed)

    if len(parameter_combinations) <= 100:
        grid_ids = np.arange(100) % len(parameter_combinations) + 1

    else:
        five_names = [name for name in names if len(parameter_ranges[name]) == 5]
        n_five = len(five_names)

        five_levels = np.array([
            [(a + j * b + t * j * j) % 5 for j in range(n_five)]
            for t in range(4) for a in range(5) for b in range(5)
        ], dtype = int)

        indices = {name: five_levels[:, j] for j, name in enumerate(five_names)}

        if "alpha" in names:
            alpha = np.append(np.repeat(np.arange(9), 11), 0)
            rng.shuffle(alpha)

            counts = np.zeros((n_five, 5, 9), dtype = int)
            for j in range(n_five):
                np.add.at(counts[j], (five_levels[:, j], alpha), 1)

            penalty = lambda c: np.maximum(2 - c, 0)**2 + np.maximum(c - 3, 0)**2
            imbalance = penalty(counts).sum()
            cols = np.arange(n_five)[:, None]

            for _ in range(5000):
                if imbalance == 0:
                    break

                i = rng.integers(0, 100, 250)
                j = rng.integers(0, 100, 250)
                ai, aj = alpha[i], alpha[j]
                vi, vj = five_levels[i].T, five_levels[j].T

                valid = (vi != vj) & (ai[None, :] != aj[None, :])

                a = counts[cols, vi, ai]
                b = counts[cols, vi, aj]
                c = counts[cols, vj, aj]
                d = counts[cols, vj, ai]

                delta = (valid * (
                    penalty(a - 1) - penalty(a) +
                    penalty(b + 1) - penalty(b) +
                    penalty(c - 1) - penalty(c) +
                    penalty(d + 1) - penalty(d)
                )).sum(axis = 0)

                chosen = int(rng.choice(np.flatnonzero(delta == delta.min())))
                i0, j0 = int(i[chosen]), int(j[chosen])
                ai0, aj0 = int(ai[chosen]), int(aj[chosen])

                if ai0 == aj0 or (delta[chosen] > 0 and rng.random() >= 0.01):
                    continue

                for f in range(n_five):
                    xi, xj = five_levels[i0, f], five_levels[j0, f]
                    if xi == xj:
                        continue

                    counts[f, xi, ai0] -= 1
                    counts[f, xi, aj0] += 1
                    counts[f, xj, aj0] -= 1
                    counts[f, xj, ai0] += 1

                alpha[i0], alpha[j0] = aj0, ai0
                imbalance += int(delta[chosen])

            if imbalance != 0:
                raise RuntimeError("Could not balance alpha across parameter levels")

            indices["alpha"] = alpha

        grid_ids = np.ravel_multi_index(
            tuple(indices[name] for name in names),
            tuple(len(parameter_ranges[name]) for name in names)
        ) + 1

    # Keep the four 25-dataset batches separate
    grid_ids = np.concatenate([
        rng.permutation(batch) for batch in grid_ids.reshape(4, 25)
    ])

    repeat_counts = {}
    samples = []

    for parameter_set, grid_id in enumerate(grid_ids, start = 1):
        grid_id = int(grid_id)
        repeat_counts[grid_id] = repeat_counts.get(grid_id, 0) + 1

        samples.append({
            "grid_id": grid_id,
            "replicate": repeat_counts[grid_id],
            "parameters": parameter_combinations[grid_id - 1].copy()
        })

    return samples

# ============================================================
# 4. Generate Task
# ============================================================

def generate_tasks(seed):

    random.seed(seed)
    np.random.seed(seed)

    map_one, map_two, *_ = task.generate_unrelated_maps()

    map_info = {
        "map_one": map_one,
        "map_two": map_two
    }

    single_generator = task.make_choice_design_file(single = True)
    single_task_map_one = single_generator.make_choice_task(
        which_map = "map_one",
        map_info = map_info
    )

    single_task_map_two = single_generator.make_choice_task(
        which_map = "map_two",
        map_info = map_info
    )

    double_generator = task.make_choice_design_file(single = False)
    double_task = double_generator.make_choice_task(
        which_map = None,
        map_info = map_info
    )

    return map_info, single_task_map_one, single_task_map_two, double_task

# convert the map to angles for Bayesian inference
def convert_maps_to_angles(map_info, map_labels):
    n_monster = task.n_monster
    position_angles = (np.arange(n_monster) * 2.0 * np.pi / n_monster)
    monster_angles = np.empty((len(map_labels), n_monster), dtype=float)

    for map_idx, map_label in enumerate(map_labels):
        map_structure = map_info[map_label]
        for position, monster in enumerate(map_structure):
            monster_angles[map_idx, monster] = position_angles[position]

    return monster_angles


# ============================================================
# 5. Task Simulation
# ============================================================

def simulate_task(model_type, parameters, parameter_set, task_type, task_sequence, map_info, seed, single_map_label = None):
    config = model_configs[model_type]
    model.validate_model_config(
        map_free = config["map_free"],
        inference = config["inference"],
        arbitration = config["arbitration"],
        memory_decay = config["memory_decay"]
    )

    h_map = fixed_hazards[task_type]["h_map"]
    h_peak = fixed_hazards[task_type]["h_peak"]

    # different choice seeds for the three tasks
    if task_type == "single":
        choice_seed = seed if single_map_label == "map_one" else seed + 1
        task_name = "choice task single ring_one" if single_map_label == "map_one" else "choice task single ring_two"
        map_labels = [single_map_label]
    else:
        choice_seed = seed + 2
        task_name = "choice task two context"
        map_labels = ["map_one", "map_two"]
    
    # set the random generator
    rng = np.random.default_rng(choice_seed)

    # get reward and map settings
    monster_angles = convert_maps_to_angles(map_info, map_labels)
    reward_matrix, _ = model.map_reward_matrix(monster_angles, n_peaks = model.n_peaks, kappa = 1.0)

    # simulation starts here
    rows = []

    for blockID, block in enumerate(task_sequence, start = 1):
    
        # initialize RW
        if config['map_free'] == "rw":
            Q_values = model.initialize_rw(task.n_monster)

        # initialize full Bayesian module
        if config["inference"] == "full_bayes":

            # initial prior is uniform
            joint_prior = model.initialize_full_bayes(n_maps = len(map_labels), n_peaks = model.n_peaks)


        # initialize reliability
        if config["arbitration"] == "pearce_hall":
            u_mf = 0.5
            u_map = 0.5
            w_mf = 1.0 - u_mf
            w_map = 1.0 - u_map

        elif config["arbitration"] == "free_weights":
            w_mf = parameters["w_mf"]
            w_map = parameters["w_map"]

        for trial in block:
            # compute values for the two options
            left_option = trial["left_choice"]
            right_option = trial["right_choice"]

            if config["map_free"] == "rw":
                mf_left_raw = Q_values[left_option]
                mf_right_raw = Q_values[right_option]
                V_mf_left = model.rescale_values(mf_left_raw)
                V_mf_right = model.rescale_values(mf_right_raw)
            else:
                mf_left_raw = np.nan
                mf_right_raw = np.nan
                V_mf_left = np.nan
                V_mf_right = np.nan

            if config["inference"] == "full_bayes":
                map_values = model.map_expected_values(joint_prior, reward_matrix)
                map_left_raw = map_values[left_option]
                map_right_raw = map_values[right_option]
                V_map_left = model.rescale_values(map_left_raw)
                V_map_right = model.rescale_values(map_right_raw)

                if config["map_free"] == "rw":
                    current_w_mf = w_mf
                    current_w_map = w_map
                    V_left = model.combine_values(mf_left_raw, map_left_raw, w_mf = current_w_mf, w_map = current_w_map)
                    V_right = model.combine_values(mf_right_raw, map_right_raw, w_mf = current_w_mf, w_map = current_w_map)
                else:
                    current_w_mf = np.nan
                    current_w_map = np.nan
                    V_left = V_map_left
                    V_right = V_map_right

            else:
                V_map_left = np.nan
                V_map_right = np.nan
                current_w_mf = np.nan
                current_w_map = np.nan
                V_left = model.combine_values(mf_left_raw)
                V_right = model.combine_values(mf_right_raw)

            # generate choices
            p_left = model.softmax_choice(V_left, V_right, parameters["beta"])
            choose_left = rng.random() < p_left

            # 0 = left, 1 = right
            model_choice = 0 if choose_left else 1

            if choose_left:
                chosen_option = left_option
                chosen_reward = trial["left_reward"]
            else:
                chosen_option = right_option
                chosen_reward = trial["right_reward"]

            row = {
                "model_type": model_type,
                "task_type": task_type, 
                "task": task_name, 
                "parameter_set": parameter_set,
                "seed": seed,
                "blockID": blockID,
                "trialID": trial["trialID"],

                "ring_label": trial["map"],
                "ring": map_info[trial["map"]], 
                "mu": trial["mu"],
                "peak_idx": trial["peak_idx"],
                "reward_dist": json.dumps(trial["reward_dist"]),

                "left_pos": trial["left_pos"],
                "right_pos": trial["right_pos"],
                "left_stim": trial["left_choice"],
                "right_stim": trial["right_choice"],
                "left_reward": trial["left_reward"],
                "right_reward": trial["right_reward"],

                "V_mf_left": V_mf_left,
                "V_mf_right": V_mf_right,
                "V_map_left": V_map_left,
                "V_map_right": V_map_right,

                "w_mf": current_w_mf,
                "w_map": current_w_map,

                "V_left": V_left,
                "V_right": V_right,

                "p_left": p_left,
                "choice": model_choice,
                "reward": chosen_reward
            }

            # add all parameter columns
            for name in list(parameter_ranges.keys()):
                if name not in row:
                    row[name] = np.nan

            for name, value in parameters.items():
                row[name] = value

            # h_map actually used in the single-map task is zero
            # if (task_type == "single" and config["inference"] == "full_bayes"):
            #     row["h_map"] = 0.0

            row["h_map"] = h_map if config["inference"] == "full_bayes" else np.nan
            row["h_peak"] = h_peak if config["inference"] == "full_bayes" else np.nan

            rows.append(row)

            # update RW after feedback
            if config["map_free"] == "rw":
                rw_result = model.rw_update(
                    Q_values = Q_values,
                    chosen_option = chosen_option,
                    reward = chosen_reward,
                    alpha = 1.0 if config["memory_decay"] else parameters["alpha"],
                    memory_decay = config["memory_decay"],
                    K = parameters.get("K")
                )

                Q_values = rw_result["Q_values"]
                rpe_mf = rw_result["rpe_mf"]

            # update Bayesian beliefs after feedback
            if config["inference"] == "full_bayes":
                rpe_map = (chosen_reward - map_values[chosen_option])
                likelihood = model.full_Bayes_feedback_likelihood(reward = chosen_reward, option_idx = chosen_option, reward_matrix = reward_matrix)
                joint_posterior, _, _, _ = model.full_Bayes_compute_posterior(joint_prior, likelihood)

                # prior for the next trial
                # joint_prior = model.full_Bayes_compute_prior(joint_posterior, h_map = h_map, h_peak = h_peak)

                # save trial-wise beliefs
                # row["joint_prior"] = json.dumps(joint_prior.tolist())
                # row["joint_posterior"] = json.dumps(joint_posterior.tolist())

                # save trial-wise beliefs
                row["joint_prior"] = json.dumps(joint_prior.tolist())
                row["joint_posterior"] = json.dumps(joint_posterior.tolist())

                # prior for the next trial
                joint_prior = model.full_Bayes_compute_prior(joint_posterior, h_map = h_map, h_peak = h_peak)

                # update reliability
                if config["arbitration"] == "pearce_hall":
                    u_mf, w_mf = model.pearce_hall_estimate(rpe = rpe_mf, uncertainty = u_mf, eta = parameters["eta"])
                    u_map, w_map = model.pearce_hall_estimate(rpe = rpe_map, uncertainty = u_map, eta = parameters["eta"])

    return pd.DataFrame(rows)

# ============================================================
# 6. Run Simulation
# ============================================================

def run_model_simulation(model_type, seed = 1, output_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/simulation_results_100_fixedbeta_fixedhazardrate", parameter_set_ids = None):
    if model_type not in model_configs:
        raise ValueError(f"Unknown model type '{model_type}'.")

    # set up parameters
    selected = sample_parameter_sets(model_type, seed = seed)

    # stratified beta sampling, balanced across four batches
    beta_rng = np.random.default_rng(seed + 100000)
    beta_quantiles = np.concatenate([beta_rng.permutation((np.arange(25) + beta_rng.random(25)) / 25) for _ in range(4)])
    beta_values = truncnorm.ppf(beta_quantiles, (1 - 4.4) / 1.7, (8 - 4.4) / 1.7, loc = 4.4, scale = 1.7)

    if parameter_set_ids is None:
        parameter_set_ids = range(1, len(selected) + 1)

    # set up output folders
    model_dir = os.path.join(output_dir, model_type)
    single_one_dir = os.path.join(model_dir, "single_map", "map_one")
    single_two_dir = os.path.join(model_dir, "single_map", "map_two")
    double_dir = os.path.join(model_dir, "double_map")

    os.makedirs(single_one_dir, exist_ok = True)
    os.makedirs(single_two_dir, exist_ok = True)
    os.makedirs(double_dir, exist_ok = True)

    # simulation loop
    for parameter_set in parameter_set_ids:
        selection = selected[parameter_set - 1]
        parameters = selection["parameters"].copy()

        if model_configs[model_type]["arbitration"] in ("free_weights", "pearce_hall"):
            parameters["beta"] = 4.4
        else:
            parameters["beta"] = float(beta_values[parameter_set - 1])

        # one reproducible seed for each simulated dataset
        simulation_seed = seed + parameter_set - 1

        map_info, single_task_one, single_task_two, double_task = generate_tasks(simulation_seed)

        # single-map task: map one
        single_one_simulation = simulate_task(
            model_type = model_type,
            parameters = parameters,
            parameter_set = parameter_set,
            task_type = "single",
            task_sequence = single_task_one,
            map_info = map_info,
            seed = simulation_seed,
            single_map_label = "map_one"
        )

        # single-map task: map two
        single_two_simulation = simulate_task(
            model_type = model_type,
            parameters = parameters,
            parameter_set = parameter_set,
            task_type = "single",
            task_sequence = single_task_two,
            map_info = map_info,
            seed = simulation_seed,
            single_map_label = "map_two"
        )

        # double-map Task
        double_simulation = simulate_task(
            model_type = model_type,
            parameters = parameters,
            parameter_set = parameter_set,
            task_type = "double",
            task_sequence = double_task,
            map_info = map_info,
            seed = simulation_seed
        )

        # save results
        file_name_single_one = (
            f"{model_type}_"
            f"single_map_one_"
            f"parameter_set_{parameter_set:06d}"
        )

        file_name_single_two = (
            f"{model_type}_"
            f"single_map_two_"
            f"parameter_set_{parameter_set:06d}"
        )

        file_name_double = (
            f"{model_type}_"
            f"double_"
            f"parameter_set_{parameter_set:06d}"
        )

        single_one_simulation.to_csv(os.path.join(single_one_dir, file_name_single_one + ".csv"), index = False)
        single_two_simulation.to_csv(os.path.join(single_two_dir, file_name_single_two + ".csv"), index = False)
        double_simulation.to_csv(os.path.join(double_dir, file_name_double + ".csv"), index = False)

        # grid id = orignal parameter id, replicate = replication times 
        metadata = {
            "model_type": model_type,
            "parameter_set": parameter_set,
            "grid_id": selection["grid_id"],
            "replicate": selection["replicate"],
            "simulation_seed": simulation_seed,

            "parameters": {
                name: float(value)
                for name, value in parameters.items()
            },

            "map_info": {
                name: [
                    int(monster)
                    for monster in map_structure
                ]
                for name, map_structure in map_info.items()
            }
        }

        if model_configs[model_type]["inference"] == "full_bayes":
            metadata["fixed_hazards"] = fixed_hazards

        file_name_metadata = (
            f"{model_type}_"
            f"parameter_set_{parameter_set:06d}"
        )

        metadata_file = os.path.join(model_dir, file_name_metadata + "_metadata.json")

        with open(metadata_file, "w") as file:
            json.dump(metadata, file, indent = 2)