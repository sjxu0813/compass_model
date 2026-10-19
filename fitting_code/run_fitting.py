#%%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from utils import model
from fitting_code.fitting import model_configs, build_model

# ============================================================
# 1. Fitting Settings
# ============================================================

# model_type = "flat_NoDecay"
# model_type = "flat_Decay"
# model_type = "hybrid_FreeWeights_NoDecay"
# model_type = "hybrid_FreeWeights_Decay"
model_type = "hybrid_Pearce_Hall_Decay"
# model_type = "hybrid_Pearce_Hall_NoDecay"

parameter_set = 2843

simulation_dir = Path("simulation_results") / model_type
output_dir = Path("fitting_results") / model_type

file_name = f"{model_type}_double_parameter_set_{parameter_set:06d}"

exp_settings = {
    "n_maps": 2,
    "n_peaks": 90,
    "n_monsters": 7,
    "Q_init": 50
}

# ============================================================
# 2. Load Data
# ============================================================

data = pd.read_csv(simulation_dir / "double_map" / f"{file_name}.csv")
config = model_configs[model_type]

# generate reward matrix for hybrid models
reward_matrix = None

if config["inference"] == "full_bayes":

    metadata_file = simulation_dir / f"{model_type}_parameter_set_{parameter_set:06d}_metadata.json"

    with open(metadata_file) as f:
        metadata = json.load(f)

    monster_angles = np.zeros((2, 7))

    for i, name in enumerate(["map_one", "map_two"]):
        monster_indices = np.asarray(metadata["map_info"][name], dtype=int)
        monster_angles[i, monster_indices] = np.arange(7) * 2.0 * np.pi / 7

    reward_matrix, _ = model.map_reward_matrix(monster_angles, n_peaks = exp_settings["n_peaks"], kappa = 1.0)

# ============================================================
# 3. Run Sampling
# ============================================================

pymc_model = build_model(model_type = model_type, data = data, reward_matrix = reward_matrix)

with pymc_model:

    trace = pm.sample(
        draws = 1000,
        tune = 1000,
        chains = 4,
        cores = 4,
        target_accept = 0.9,
        random_seed = 42,
        progressbar="combined+stats"
    )

# ============================================================
# 4. Save Results
# ============================================================

output_dir.mkdir(parents = True, exist_ok = True)
trace.to_netcdf(output_dir / f"{file_name}_posterior.nc")
summary = az.summary(trace, var_names = config["parameters"])
summary.to_csv(output_dir / f"{file_name}_summary.csv")

# ============================================================
# 5. Check Results
# ============================================================

print(summary)

print("\nGenerating vs. estimated parameters:")

for name in config["parameters"]:
    true_value = data[name].iloc[0]
    estimated_value = summary.loc[name, "mean"]

    print(f"{name}: true = {true_value:.3f}, estimated = {estimated_value:.3f}")

print("\nDivergences:", int(trace.sample_stats["diverging"].sum()))

# %%
