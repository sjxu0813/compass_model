import numpy as np
from scipy.stats import vonmises, norm

# ============================================================
# 1. Model Initialization
# ============================================================

# parameter initialization
reward_min = 0.0
reward_max = 100.0
Q_init = 50.0
decay_target = 50.0
n_maps = 2
n_peaks = 12
n_monsters = 7
uncertainty = 0.5

# model configuration validation
## UNDER DEVELOPMENT!!!! adding more models in the future
implemented_map_free = ["rw", None]
implemented_inference = ["full_bayes"]
implemented_arbitration = ["pearce_hall", "free_weights", None]

def validate_model_config(map_free = "rw", inference = None, arbitration = None, memory_decay = False):

    if map_free is not None and map_free not in implemented_map_free:
        raise ValueError(
            f"Map-free model '{map_free}' is not implemented. "
            f"Available models: {implemented_map_free}"
        )

    if inference is not None and inference not in implemented_inference:
        raise ValueError(
            f"Inference model '{inference}' is not implemented. "
            f"Available models: {implemented_inference}"
        )

    if arbitration is not None and arbitration not in implemented_arbitration:
        raise ValueError(
            f"Arbitration model '{arbitration}' is not implemented. "
            f"Available models: {implemented_arbitration}"
        )

    if not isinstance(memory_decay, (bool, np.bool_)):
        raise TypeError(
            "'memory_decay' must be True or False."
        )

    model_name = "_".join([
        map_free if map_free is not None else "none",
        inference if inference is not None else "no_inference",
        arbitration if arbitration is not None else "no_arbitration",
        "decay" if memory_decay else "no_decay",
    ])

    return model_name


# ============================================================
# 2. Rascorla-Wagner Module
# ============================================================

# initialize RW values for all options
def initialize_rw(n_options, Q_init = 50.0):
    Q_values = np.full(n_options, Q_init, dtype = "float64")

    return Q_values

# update Q values using the Rescorla-Wagner rule
def rw_update(Q_values, chosen_option, reward, alpha, memory_decay = False, K = None, decay_target = 50.0):
    """
    Parameters
    ----------
    Q_values: np.ndarray
        Q values of all options

    chosen_option: int
        Index of the chosen option

    reward: float
        Received reward

    alpha: float
        RW learning rate

    memory_decay: bool
        Whether memory decay is enabled

    K: float or None
        Memory decay rate

    decay_target: float
        Value toward which unchosen options decay

    Returns
    -------
    dict
        Q_values, predicted_reward, rpe_mf
    """

    Q_values = Q_values.copy()

    # reward prediction from the chosen option, based Q values from the last trials
    predicted_reward = Q_values[chosen_option]

    # reward prediction error
    rpe_mf = reward - predicted_reward

    # memory decay for unchosen and unseen options if memory_decay is enabled
    if memory_decay:

        if K is None:
            raise ValueError("'K' must be provided when memory_decay is True.")

        unchosen = np.arange(len(Q_values)) != chosen_option
        
        # update Q values for the chosen option, alpha = 1
        Q_values[chosen_option] += rpe_mf

        # update Q values for unchosen and unseen options
        Q_values[unchosen] += (K * (decay_target - Q_values[unchosen]))

    else:
        # update Q values for the chosen option, controlled by alpha
        Q_values[chosen_option] += alpha * rpe_mf

    return {"Q_values": Q_values, "predicted_reward": predicted_reward, "rpe_mf": rpe_mf}

# ============================================================
# 3. Inference Module
# ============================================================

# initialize a uniform joint belief over map identity and peak position
def initialize_full_bayes(n_maps = n_maps, n_peaks = n_peaks):
    joint_posterior = np.full((n_maps, n_peaks), 1.0 / (n_maps * n_peaks), dtype = "float64")

    return joint_posterior

# peak change probabilities
def peak_change_matrix(n_peaks = n_peaks):
    transition = np.ones((n_peaks, n_peaks), dtype = "float64")
    np.fill_diagonal(transition, 0)
    transition /= n_peaks - 1
    return transition

# compute the joint prior for the next trial
def full_Bayes_compute_prior(joint_posterior, h_map,h_peak):
    """
    Parameters
    ----------
    joint_posterior : np.ndarray
        Previous joint posterior p(map, peak)
        Shape: (n_maps, n_peaks)

    h_map : float
        Probability that the map/context changes

    h_peak : float
        Probability that the peak changes,
        conditional on the map remaining the same

    Returns
    -------
    joint_prior : np.ndarray
        Prior p(map, peak) for the next trial
    """

    if not 0.0 <= h_map <= 1.0:
        raise ValueError("'h_map' must be between 0 and 1.")

    if not 0.0 <= h_peak <= 1.0:
        raise ValueError("'h_peak' must be between 0 and 1.")

    joint_posterior = np.asarray(joint_posterior, dtype = "float64")

    n_maps, n_peaks = (joint_posterior.shape)
    peak_change = peak_change_matrix(n_peaks = n_peaks)
    identity = np.eye(n_peaks)

    same_map_transition = ((1.0 - h_peak) * identity + h_peak * peak_change)
    joint_prior = np.zeros_like(joint_posterior)

    for previous_map in range(n_maps):
        joint_prior[previous_map] += ((1.0 - h_map) * (joint_posterior[previous_map] @ same_map_transition))

        for new_map in range(n_maps):
            if new_map == previous_map:
                continue
            joint_prior[new_map] += (h_map * (joint_posterior[previous_map] @ peak_change))

    return joint_prior

# generate expected rewards for every combination of map, peak and monster
def map_reward_matrix(map_monsters, n_peaks, kappa = 1.0):
    """
    Parameters
    ----------
    map_monsters : np.ndarray
        Monster positions under each map, shape: (n_maps, n_monsters)
        Each entry should be an angle in radians between 0 and 2*pi

    n_peaks : int
        Number of candidate peak positions around the map

    kappa : float
        Von Mises concentration parameter, fixed at 1 in the current model.

    Returns
    -------
    reward_matrix : np.ndarray
        Shape:
            (n_maps, n_peaks, n_monsters)

    peaks : np.ndarray
        Possible peak positions in radians.
    """

    map_monsters = np.asarray(map_monsters, dtype = "float64")
    n_maps, n_monsters = (map_monsters.shape)

    peaks = np.linspace(0.0, 2.0 * np.pi, n_peaks, endpoint = False)

    reward_matrix = np.zeros((n_maps, n_peaks, n_monsters), dtype = "float64")

    for map_idx in range(n_maps):
        for peak_idx, mu in enumerate(peaks):

            pdf_vals = vonmises.pdf(map_monsters[map_idx], loc = mu, kappa = kappa)
            min_val = vonmises.pdf(mu + np.pi, loc = mu, kappa = kappa)
            max_val = vonmises.pdf(mu, loc = mu, kappa = kappa)

            rewards = np.round(100.0 * (pdf_vals - min_val) / (max_val - min_val))
            reward_matrix[map_idx, peak_idx] = rewards

    return reward_matrix, peaks

# map-based expected reward of every monster before observing the current feedback
def map_expected_values(joint_prior, reward_matrix):
    expected_values = np.sum(joint_prior[:, :, None] * reward_matrix, axis = (0, 1))

    return expected_values

# calculate the likelihood of the observed feedback under every map x peak pairs
def full_Bayes_feedback_likelihood(reward, option_idx, reward_matrix, obs_sd = 5.0):
    expected_reward = (reward_matrix[:,:,option_idx,])
    likelihood = norm.pdf(reward, loc = expected_reward, scale = obs_sd)

    return likelihood

# compute the posteriors
def full_Bayes_compute_posterior(joint_prior, likelihood):
    """
    Parameters
    ----------
    joint_prior : np.ndarray
        Prior probability over map and peak
        Shape: (n_maps, n_peaks)

    likelihood : np.ndarray
        Feedback likelihood under each map and peak
        Shape: (n_maps, n_peaks)

    Returns
    -------
    joint_posterior : np.ndarray
        Posterior probability over map and peak

    map_posterior : np.ndarray
        Posterior probability of each map

    peak_posterior : np.ndarray
        Posterior probability of each peak, marginalized over maps

    peak_given_map_posterior : np.ndarray
        Posterior probability of each peak conditional on each map
    """

    unnormalized_posterior = joint_prior * likelihood
    normalizer = unnormalized_posterior.sum()

    if normalizer <= 0 or not np.isfinite(normalizer):
        raise ValueError(
            "Bayesian posterior could not be normalized because the normalizer is non-positive or non-finite."
        )

    # Compute the normalized joint posterior
    joint_posterior = unnormalized_posterior / normalizer

    # p(map | feedback)
    map_posterior = joint_posterior.sum(axis = 1)

    # p(peak | feedback)
    peak_posterior = joint_posterior.sum(axis = 0)

    # p(peak | map, feedback)
    peak_given_map_posterior = np.divide(
        joint_posterior,
        map_posterior[:, None],
        out = np.zeros_like(joint_posterior),
        where = map_posterior[:, None] > 0
    )

    return joint_posterior, map_posterior, peak_posterior, peak_given_map_posterior

# ============================================================
# 4. Reliability Estimate
# ============================================================

# Pearce-Hall-style estimator
def pearce_hall_estimate(rpe, uncertainty, eta, reward_range = reward_max):
    """
    Parameters
    ----------
    rpe : float
        Reward prediction error from the decision system

    uncertainty : float
        Uncertainty is estimated based on unsigned prediction error, init value is 0.5

    eta : float
        Reliability learning rate between 0 and 1, same for two systems

    reward_range : float
        Maximum possible reward difference, fixed at 100 for the current task

    Returns
    -------
    uncertainty : float
        Uncertainty estimate between 0 and 1

    reliability : float
        Reliability (w) between 0 and 1
    """

    if not 0.0 <= eta <= 1.0:
        raise ValueError("'eta' must be between 0 and 1.")

    normalized_error = abs(rpe) / reward_range

    uncertainty = (1.0 - eta) * uncertainty + eta * normalized_error
    reliability = 1.0 - uncertainty

    return uncertainty, reliability

# ============================================================
# 5. Generative Model
# ============================================================

# rescale values from the task reward range to [0, 1].
def rescale_values(value, value_min = reward_min, value_max = reward_max):
    scaled_values = (value - value_min) / (value_max - value_min)

    return scaled_values

# combine map-free and map-based Q values
def combine_values(mf_value = None, map_value = None, w_mf = None, w_map = None):

    # RW model
    if map_value is None:
        mf_value = rescale_values(mf_value)
        V_value = mf_value
        return V_value

    # full Bayesian model
    if mf_value is None:
        map_value = rescale_values(map_value)
        V_value = map_value
        return V_value

    # hybrid models
    if w_mf is None or w_map is None:
        raise ValueError("'w_mf' and 'w_map' are required for hybrid models")

    mf_value = rescale_values(mf_value)
    map_value = rescale_values(map_value)
    V_value = w_mf * mf_value + w_map * map_value

    return V_value

# calculate the probability of choosing the left option.
def softmax_choice(V_left, V_right, beta):
    V_diff = V_left - V_right
    choice_prob = 1.0 / (1.0 + np.exp(-beta * V_diff))

    return choice_prob




    