#%%
import numpy as np
import pytensor.tensor as pt
from pytensor import scan
from pytensor.raise_op import Assert
from utils import model
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
        "parameters": ["alpha", "beta", "K"]
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
        "parameters": ["alpha", "beta", "h_map", "h_peak", "w_mf", "w_map", "K"]
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
        "parameters": ["alpha", "beta", "h_map", "h_peak", "eta", "K"]
    }
}

exp_settings = {
    "n_maps": 2,
    "n_peaks": 90,
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
    peak_change = pt.as_tensor_variable(model.peak_change_matrix(n_peaks = n_peaks, n_monsters = n_monsters))
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

    alpha = pt.as_tensor_variable(parameters["alpha"])
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

        pm.Potential("choice_loglikelihood", pt.sum(logp))

    return model

