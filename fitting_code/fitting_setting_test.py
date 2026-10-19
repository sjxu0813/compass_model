#%%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytensor

from utils import model
from fitting_code.fitting import model_configs, choice_loglikelihood, build_model
#%%
output_dir = Path("simulation_results")  # change if necessary

for model_type, config in model_configs.items():
    model_dir = output_dir / model_type
    data = pd.read_csv(model_dir / "double_map" / f"{model_type}_double_parameter_set_000001.csv")

    with open(model_dir / f"{model_type}_parameter_set_000001_metadata.json") as f:
        metadata = json.load(f)

    parameters = {
        name: float(data[name].iloc[0])
        for name in config["parameters"]
    }

    reward_matrix = None

    if config["inference"] == "full_bayes":
        monster_angles = np.zeros((2, 7))

        for i, name in enumerate(["map_one", "map_two"]):
            monster_indices = np.asarray(metadata["map_info"][name], dtype=int)
            monster_angles[i, monster_indices] = np.arange(7) * 2 * np.pi / 7

        reward_matrix, _ = model.map_reward_matrix(
            monster_angles, n_peaks=90, kappa=1.0
        )

    p_left, logp = choice_loglikelihood(
        model_type, parameters,
        data["left_choice"], data["right_choice"],
        data["model_choice"],
        data["left_reward"], data["right_reward"],
        data["blockID"], reward_matrix
    )

    fitted_p, fitted_logp = pytensor.function([], [p_left, logp])()

    simulated_p = data["p_left"].to_numpy()

    expected_logp = np.where(
        data["model_choice"].to_numpy() == 0,
        np.log(simulated_p),
        np.log1p(-simulated_p)
    )

    np.testing.assert_allclose(fitted_p, simulated_p, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fitted_logp, expected_logp, rtol=0, atol=1e-10)

    print(f"{model_type}: numerical test passed")

    # Test PyMC compilation and gradients on short data
    test_data = data.groupby("blockID", sort=False).head(3)

    pymc_model = build_model(model_type, test_data, reward_matrix)

    point = pymc_model.initial_point()
    assert np.isfinite(pymc_model.compile_logp()(point))
    assert np.all(np.isfinite(pymc_model.compile_dlogp()(point)))

    print(f"{model_type}: PyMC test passed")

# %%
