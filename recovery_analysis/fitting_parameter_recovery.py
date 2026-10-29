#%%
import os
import sys
import json
import pandas as pd
import arviz as az
import numpy as np
import pytensor.tensor as pt
from pytensor import scan
from pytensor.raise_op import Assert
import model
import pymc as pm

# %%
# ============================================================
# 1. Model Configuration
# ============================================================

model_configs = {
    "flat_NoDecay": {
        "map_free": "rw",
        "inference": None,
        "arbitration": None,
        "memory_decay": False,
        "parameters": ["alpha", "beta"]
    },
    "flat_Decay": {
        "map_free": "rw",
        "inference": None,
        "arbitration": None,
        "memory_decay": True,
        "parameters": ["beta", "K"]
    },
    "hybrid_FreeWeights_NoDecay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "free_weights",
        "memory_decay": False,
        "parameters": ["alpha", "beta", "h_map", "h_peak", "w_mf", "w_map"]
    },
    "hybrid_FreeWeights_Decay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "free_weights",
        "memory_decay": True,
        "parameters": ["beta", "h_map", "h_peak", "w_mf", "w_map", "K"]
    },
    "hybrid_Pearce_Hall_NoDecay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "pearce_hall",
        "memory_decay": False,
        "parameters": ["alpha", "beta", "h_map", "h_peak", "eta"]
    },
    "hybrid_Pearce_Hall_Decay": {
        "map_free": "rw",
        "inference": "full_bayes",
        "arbitration": "pearce_hall",
        "memory_decay": True,
        "parameters": ["beta", "h_map", "h_peak", "eta", "K"]
    },
    "fullBayes_NoDecay": {
        "map_free": None,
        "inference": "full_bayes",
        "arbitration": None,
        "memory_decay": False,
        "parameters": ["beta", "h_map", "h_peak"]
    },
}

exp_settings = {
    "n_maps": 2,
    "n_peaks": 12,
    "n_monsters": 7,
    "Q_init": 50
}

# ============================================================
# 2. Rescorla-Wagner Module
# ============================================================

def initialize_rw(n_options, Q_init):
    Q_values = pt.full((n_options,), Q_init, dtype = "float64")
    return Q_values

def rw_update(Q_values, chosen_option, reward, alpha, memory_decay = False, K = None, decay_target = 50.0):

    predicted_reward = Q_values[chosen_option]
    rpe_mf = reward - predicted_reward

    Q_values = pt.set_subtensor(Q_values[chosen_option], predicted_reward + alpha * rpe_mf)

    if memory_decay:
        if K is None:
            raise ValueError("'K' must be provided when memory_decay = True.")

        unchosen = pt.neq(pt.arange(Q_values.shape[0]), chosen_option)
        Q_values = pt.where(unchosen, Q_values + K * (decay_target - Q_values), Q_values)
    
    return {
        "Q_values": Q_values,
        "predicted_reward": predicted_reward,
         "rpe_mf": rpe_mf
    }

# ============================================================
# 3. Inference Module
# ============================================================

# initialize uniform beliefs over map identity and peak position
def initialize_full_bayes(n_maps = exp_settings["n_maps"], n_peaks = exp_settings["n_peaks"]):
    joint_prior = pt.full((n_maps, n_peaks), 1.0 / (n_maps * n_peaks), dtype = "float64")
    return joint_prior

# compute the joint prior for the next trial
def full_Bayes_compute_prior(joint_posterior, h_map, h_peak):

    n_peaks = exp_settings["n_peaks"]
    n_monsters = exp_settings["n_monsters"]

    # peak transition matrix
    peak_change = pt.as_tensor_variable(model.peak_change_matrix(n_peaks = n_peaks))
    identity = pt.eye(exp_settings["n_peaks"], dtype = "float64")

    # potential peak change within the same map
    same_map_transition = ((1.0 - h_peak) * identity + h_peak * peak_change)

    # update two maps
    prior_map_1 = (1.0 - h_map) * pt.dot(joint_posterior[0], same_map_transition) + h_map * pt.dot(joint_posterior[1], peak_change)
    prior_map_2 = ((1.0 - h_map) * pt.dot(joint_posterior[1], same_map_transition) + h_map * pt.dot(joint_posterior[0], peak_change))

    joint_prior = pt.stack([prior_map_1, prior_map_2], axis = 0)

    return joint_prior

# expected rewards before observing current feedback
def map_expected_values(joint_prior, reward_matrix):
    reward_matrix = pt.as_tensor_variable(reward_matrix)
    expected_values = pt.sum(joint_prior[:, :, None] * reward_matrix, axis = (0, 1))

    return expected_values

# feedback likelihood under each map and peak combination
def full_Bayes_feedback_likelihood(reward, option_idx, reward_matrix, obs_sd = 5.0):
    reward_matrix = pt.as_tensor_variable(reward_matrix)
    expected_reward = reward_matrix[:, :, option_idx]
    likelihood = pt.exp(-0.5 * pt.square((reward - expected_reward) / obs_sd))/ (obs_sd * np.sqrt(2.0 * np.pi))

    return likelihood

# compute Bayesian posterior distributions
def full_Bayes_compute_posterior(joint_prior, likelihood):

    unnormalized_posterior = joint_prior * likelihood
    normalizer = pt.sum(unnormalized_posterior)

    normalizer = Assert(
        "Bayesian posterior normalizer must be positive and finite"
    )(
        normalizer,
        pt.gt(normalizer, 0),
        ~pt.isnan(normalizer),
        ~pt.isinf(normalizer)
    )

    # joint posterior
    joint_posterior = unnormalized_posterior / normalizer

    return joint_posterior

# ============================================================
# 4. Reliability Estimate
# ============================================================

def pearce_hall_estimate(rpe, uncertainty, eta, reward_range = 100.0):
    normalized_error = pt.abs(rpe) / reward_range
    uncertainty = (1.0 - eta) * uncertainty + eta * normalized_error

    return uncertainty

# ============================================================
# 5. Trialwise Choice Likelihood
# ============================================================  

def choice_loglikelihood(model_type, parameters, left_choice, right_choice, model_choice, left_reward, right_reward, block_id, reward_matrix = None):

    config = model_configs[model_type]
    n_monsters = exp_settings["n_monsters"]
    Q_init = exp_settings["Q_init"]

    left_choice = pt.as_tensor_variable(np.asarray(left_choice, dtype = "int64"))
    right_choice = pt.as_tensor_variable(np.asarray(right_choice, dtype = "int64"))
    model_choice = pt.as_tensor_variable(np.asarray(model_choice, dtype = "int64"))
    left_reward = pt.as_tensor_variable(np.asarray(left_reward, dtype = "float64"))
    right_reward = pt.as_tensor_variable(np.asarray(right_reward, dtype = "float64"))
    block_id = pt.as_tensor_variable(np.asarray(block_id))

    # observed choice: 0 = left, 1 = right
    chosen_reward = pt.where(pt.eq(model_choice, 0), left_reward, right_reward)

    # identify the first trial of each block
    block_start = pt.concatenate([pt.ones((1,), dtype="bool"), pt.neq(block_id[1:], block_id[:-1])])

    alpha = pt.as_tensor_variable(parameters["alpha"] if "alpha" in parameters else 1.0)
    beta = pt.as_tensor_variable(parameters["beta"])
    K = pt.as_tensor_variable(parameters["K"]) if config["memory_decay"] else pt.as_tensor_variable(0.0)

    initial_q = initialize_rw(n_options = n_monsters, Q_init = Q_init)
    sequences = [left_choice, right_choice, model_choice, chosen_reward, block_start]

    # flat models
    if config["inference"] is None:

        def flat_trial_step(left, right, choice, reward, reset, q_prev, alpha, beta, K):

            # reset flat models at the beginning of each block
            Q_values = pt.where(reset, initialize_rw(n_options = n_monsters, Q_init = Q_init), q_prev)

            # compute values
            V_left = Q_values[left] / 100.0
            V_right = Q_values[right] / 100.0
            logit = beta * (V_left - V_right)
            p_left = pt.sigmoid(logit)
            logp = pt.where(pt.eq(choice, 0), -pt.softplus(-logit), -pt.softplus(logit))

            # update flat models after feedback
            chosen_option = pt.where(pt.eq(choice, 0), left, right)

            rw = rw_update(Q_values, chosen_option, reward, alpha, memory_decay = config["memory_decay"], K = K)

            return rw["Q_values"], p_left, logp

        (_, p_left, logp), _ = scan(
            fn = flat_trial_step,
            sequences = sequences,
            outputs_info = [initial_q, None, None],
            non_sequences = [alpha, beta, K],
            strict = True
        )

        return p_left, logp

    # hybrid models
    reward_matrix = pt.as_tensor_variable(np.asarray(reward_matrix, dtype = "float64"))

    h_map = pt.as_tensor_variable(parameters["h_map"])
    h_peak = pt.as_tensor_variable(parameters["h_peak"])

    # full Bayesian model
    if config["map_free"] is None:

        def bayes_trial_step(left, right, choice, reward, reset, prior_prev, beta, h_map, h_peak, reward_matrix):

            joint_prior = pt.where(reset, initialize_full_bayes(), prior_prev)

            # compute expected rewards
            map_values = map_expected_values(joint_prior, reward_matrix)

            # compute choice probability and log-likelihood
            logit = beta * ((map_values[left] - map_values[right]) / 100.0)
            p_left = pt.sigmoid(logit)
            logp = pt.where(pt.eq(choice, 0), -pt.softplus(-logit), -pt.softplus(logit))

            # identify chosen monster
            chosen_option = pt.where(pt.eq(choice, 0), left, right)

            # update Bayesian beliefs after feedback
            likelihood = full_Bayes_feedback_likelihood(reward, chosen_option, reward_matrix, obs_sd = 5.0)
            joint_posterior = full_Bayes_compute_posterior(joint_prior, likelihood)
            next_prior = full_Bayes_compute_prior(joint_posterior, h_map, h_peak)

            return next_prior, p_left, logp

        (_, p_left, logp), _ = scan(
            fn = bayes_trial_step,
            sequences = sequences,
            outputs_info = [initialize_full_bayes(), None, None],
            non_sequences = [beta, h_map, h_peak, reward_matrix],
            strict = True
        )

        return p_left, logp

    pearce_hall = config["arbitration"] == "pearce_hall"

    eta = pt.as_tensor_variable(parameters["eta"]) if pearce_hall else pt.as_tensor_variable(0.0)
    w_mf_fixed = pt.as_tensor_variable(parameters["w_mf"]) if not pearce_hall else pt.as_tensor_variable(0.5)
    w_map_fixed = pt.as_tensor_variable(parameters["w_map"]) if not pearce_hall else pt.as_tensor_variable(0.5)

    def hybrid_trial_step(left, right, choice, reward, reset, q_prev, prior_prev, u_mf_prev, u_map_prev, alpha, beta, K, h_map, h_peak, eta, w_mf_fixed, w_map_fixed, reward_matrix):

        # reset all learned states at the beginning of each block
        Q_values = pt.where(reset, initialize_rw(n_options = n_monsters, Q_init = Q_init), q_prev)
        joint_prior = pt.where(reset, initialize_full_bayes(), prior_prev)
        u_mf = pt.where(reset, 0.5, u_mf_prev)
        u_map = pt.where(reset, 0.5, u_map_prev)

        # compute expected rewards
        map_values = map_expected_values(joint_prior, reward_matrix)

        # arbitration weights
        w_mf = (1.0 - u_mf) if pearce_hall else w_mf_fixed
        w_map = (1.0 - u_map) if pearce_hall else w_map_fixed

        # combine RW and map-based values
        V_left = w_mf * (Q_values[left] / 100.0) + w_map * (map_values[left] / 100.0)
        V_right = w_mf * (Q_values[right] / 100.0) + w_map * (map_values[right] / 100.0)

        # compute choice probability and log-likelihood
        logit = beta * (V_left - V_right)
        p_left = pt.sigmoid(logit)

        logp = pt.where(pt.eq(choice, 0), -pt.softplus(-logit), -pt.softplus(logit))

        # identify chosen monster
        chosen_option = pt.where(pt.eq(choice, 0), left, right)

        # update RW after feedback
        rw = rw_update(Q_values, chosen_option, reward, alpha, memory_decay = config["memory_decay"], K = K)

        # update Bayesian beliefs after feedback
        rpe_map = reward - map_values[chosen_option]
        likelihood = full_Bayes_feedback_likelihood(reward, chosen_option, reward_matrix, obs_sd = 5.0)
        joint_posterior = full_Bayes_compute_posterior(joint_prior, likelihood)
        next_prior = full_Bayes_compute_prior(joint_posterior, h_map, h_peak)

        # Update Pearce-Hall reliability
        if pearce_hall:
            u_mf = pearce_hall_estimate(rw["rpe_mf"], u_mf, eta)
            u_map = pearce_hall_estimate(rpe_map, u_map, eta)

        return rw["Q_values"], next_prior, u_mf, u_map, p_left, logp

    (_, _, _, _, p_left, logp), _ = scan(
        fn = hybrid_trial_step,
        sequences = sequences,
        outputs_info = [initial_q, initialize_full_bayes(), pt.constant(0.5, dtype="float64"), pt.constant(0.5, dtype="float64"), None, None],
        non_sequences = [alpha, beta, K, h_map, h_peak, eta, w_mf_fixed, w_map_fixed, reward_matrix],
        strict = True
    )

    return p_left, logp

# ============================================================
# 5. Model Constrcution
# ============================================================  

def recover_beliefs(data, reward_matrix, h_map, h_peak):

    priors = []
    posteriors = []
    previous_block = None

    for t, row in enumerate(data.itertuples(index = False)):

        # reset beliefs at each block
        if t == 0 or row.blockID != previous_block:
            joint_prior = model.initialize_full_bayes(
                n_maps = exp_settings["n_maps"],
                n_peaks = exp_settings["n_peaks"]
            )

        priors.append(joint_prior.copy())

        # observed choice and reward
        chosen_option = int(row.left_choice if row.model_choice == 0 else row.right_choice)
        reward = row.left_reward if row.model_choice == 0 else row.right_reward

        # Bayesian update
        likelihood = model.full_Bayes_feedback_likelihood(reward, chosen_option, reward_matrix, obs_sd = 5.0)
        joint_posterior, _, _, _ = model.full_Bayes_compute_posterior(joint_prior, likelihood)
        posteriors.append(joint_posterior.copy())

        # prior for the next trial
        joint_prior = model.full_Bayes_compute_prior(joint_posterior, h_map, h_peak)

        previous_block = row.blockID

    return np.stack(priors), np.stack(posteriors)

def build_model(model_type, data, reward_matrix = None):
    config = model_configs[model_type]
    with pm.Model() as model:

        # set parameter priors
        parameters = {}
        for name in config["parameters"]:
            if name == "beta":
                parameters[name] = pm.TruncatedNormal(name, mu = 4.4, sigma = 1.7, lower = 1.0, upper = 8.0)
            else:
                parameters[name] = pm.Uniform(name, lower = 0.0, upper = 1.0)

        # compute trial-wise choice likelihood
        p_left, logp = choice_loglikelihood(
            model_type = model_type,
            parameters = parameters,
            left_choice = data["left_choice"],
            right_choice = data["right_choice"],
            model_choice = data["model_choice"],
            left_reward = data["left_reward"],
            right_reward = data["right_reward"],
            block_id = data["blockID"],
            reward_matrix=reward_matrix
        )

        pm.Deterministic("trial_loglik", logp)
        pm.Potential("choice_loglikelihood", pt.sum(logp))

    return model

# ============================================================
# 6. Run Model Fitting
# ============================================================  

if __name__ == "__main__":
    # fit all models
    model_types = list(model_configs)

    # divide datasets across SLURM array tasks; run everything when started locally.
    task_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    current_index = 0
    selected = 0

    for model_type in model_types:

        simulation_dir = os.path.join("simulation_results", model_type)
        data_dir = os.path.join(simulation_dir, "double_map")
        output_dir = os.path.join("fitting_results", model_type)
        config = model_configs[model_type]

        os.makedirs(output_dir, exist_ok = True)

        prefix = f"{model_type}_double_parameter_set_"
        csv_files = sorted(f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".csv") and f[len(prefix):-4].isdigit())

        for csv_file in csv_files:

            # select this SLURM task's share, without changing the nested loops.
            if current_index % n_tasks != task_index:
                current_index += 1
                continue
            current_index += 1
            selected += 1

            file_name = os.path.splitext(csv_file)[0]
            parameter_set = int(file_name[len(prefix):])
            data = pd.read_csv(os.path.join(data_dir, csv_file))

            posterior_path = os.path.join(output_dir, f"{file_name}_posterior.nc")
            summary_path = os.path.join(output_dir, f"{file_name}_summary.csv")
            belief_dir = os.path.join(output_dir, "beliefs")
            belief_path = os.path.join(belief_dir, f"parameter_set_{parameter_set:06d}_beliefs.npz")

            expected_outputs = [posterior_path, summary_path]
            if config["inference"] == "full_bayes":
                expected_outputs.append(belief_path)

            # skip fully saved fits when restarting interrupted SLURM arrays.
            if all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in expected_outputs):
                print(f"Skipping completed: {model_type}, parameter set {parameter_set}", flush = True)
                continue

            # generate reward matrix for Bayesian inference models
            reward_matrix = None

            if config["inference"] == "full_bayes":
                metadata_file = os.path.join(simulation_dir, f"{model_type}_parameter_set_{parameter_set:06d}_metadata.json")

                with open(metadata_file) as f:
                    metadata = json.load(f)

                monster_angles = np.zeros((exp_settings["n_maps"], exp_settings["n_monsters"]))

                for i, name in enumerate(["map_one", "map_two"]):
                    monster_indices = np.asarray(metadata["map_info"][name], dtype = int)
                    monster_angles[i, monster_indices] = np.arange(exp_settings["n_monsters"]) * 2.0 * np.pi / exp_settings["n_monsters"]

                reward_matrix, _ = model.map_reward_matrix(monster_angles, n_peaks = exp_settings["n_peaks"], kappa = 1.0)

            pymc_model = build_model(model_type = model_type, data = data, reward_matrix = reward_matrix)

            with pymc_model:
                trace = pm.sample(
                    draws = 1000,
                    tune = 1000,
                    chains = 4,
                    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                    target_accept = 0.9,
                    random_seed = 1
                )

                trace.add_groups(log_likelihood = trace.posterior[["trial_loglik"]].rename({"trial_loglik": "choice"}))

            # save fitted parameters; use temporary filenames so incomplete files
            # cannot be mistaken for finished fits after an interrupted job.
            trace.posterior.attrs["model_type"] = model_type
            trace.posterior.attrs["parameter_set"] = parameter_set
            tmp_posterior = f"{posterior_path}.part.{os.getpid()}.nc"
            trace.to_netcdf(tmp_posterior)
            os.replace(tmp_posterior, posterior_path)

            summary = az.summary(trace, var_names = config["parameters"])
            summary.insert(0, "model_type", model_type)
            summary.insert(1, "parameter_set", parameter_set)
            tmp_summary = f"{summary_path}.part.{os.getpid()}.csv"
            summary.to_csv(tmp_summary)
            os.replace(tmp_summary, summary_path)

            # save trial-wise Bayesian beliefs using posterior-mean hazard estimates
            if config["inference"] == "full_bayes":

                h_map_hat = float(trace.posterior["h_map"].mean().item())
                h_peak_hat = float(trace.posterior["h_peak"].mean().item())
                joint_prior, joint_posterior = recover_beliefs(data, reward_matrix, h_map_hat, h_peak_hat)

                os.makedirs(belief_dir, exist_ok = True)
                tmp_beliefs = f"{belief_path}.part.{os.getpid()}.npz"

                np.savez_compressed(
                    tmp_beliefs,
                    model_type = model_type,
                    parameter_set = parameter_set,
                    joint_prior = joint_prior,
                    joint_posterior = joint_posterior,
                    blockID = data["blockID"].to_numpy(),
                    h_map = h_map_hat,
                    h_peak = h_peak_hat
                )
                os.replace(tmp_beliefs, belief_path)