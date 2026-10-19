import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# set path
data_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/simulation_results"
output_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/model_behavior_analysis"
os.makedirs(output_dir, exist_ok=True)

# compute accuracy
def compute_accuracy(data):
    data = data.copy()

    data["chosen_reward"] = np.where(
        data["model_choice"] == 0,
        data["left_reward"],
        data["right_reward"]
    )

    data["best_reward"] = data[["left_reward", "right_reward"]].max(axis=1)
    data["accuracy"] = (data["chosen_reward"] == data["best_reward"]).astype(float)

    return data

# extract trials around changes
def extract_change_windows(data, task_label, window = 5):
    rows = []
    event_id = 0

    # changes are detected separately within each block
    for block_id, block in data.groupby("blockID"):
        block = block.sort_values("trialID").reset_index(drop = True)

        previous_peak = block["peak_idx"].shift(1)
        previous_map = block["map"].shift(1)

        peak_change = ((block["peak_idx"] != previous_peak) & previous_peak.notna())

        if task_label in ["single_map_one", "single_map_two"]:
            events = [(position, "peak_change") for position in np.where(peak_change)[0]]
        else:
            context_change = ((block["map"] != previous_map) & previous_map.notna())
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

# process simulation data
overall_results = []
change_results = []

files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive = True)


# process each simulation file
def process_file(file):
    data = pd.read_csv(file)
    data = compute_accuracy(data)
    model_type = data["model_type"].iloc[0]
    parameter_set = data["parameter_set"].iloc[0]
    task_type = data["task_type"].iloc[0]

    # identify the three task types
    if task_type == "single":
        map_label = data["map"].iloc[0]
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
        parameter_summary = change_data.groupby(["change_type", "relative_trial"], as_index = False).agg(accuracy=("accuracy", "mean"), n_events=("event_id", "nunique"))
        parameter_summary["model_type"] = model_type
        parameter_summary["parameter_set"] = parameter_set
        parameter_summary["task"] = task_label

    return overall_result, parameter_summary


if __name__ == "__main__":

    files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive = True)
    print(f"Found {len(files)} simulation files")

    overall_results = []
    change_results = []

    # parallel processing
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    with ProcessPoolExecutor(max_workers = n_workers) as executor:
        for overall_result, parameter_summary in executor.map(process_file, files):
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


# for file_index, file in enumerate(files, start = 1):
#     data = pd.read_csv(file)
#     data = compute_accuracy(data)
#     model_type = data["model_type"].iloc[0]
#     parameter_set = data["parameter_set"].iloc[0]
#     task_type = data["task_type"].iloc[0]

#     # identify the three task types
#     if task_type == "single":

#         map_label = data["map"].iloc[0]

#         if map_label == "map_one":
#             task_label = "single_map_one"
#         else:
#             task_label = "single_map_two"
#     else:
#         task_label = "double_map"

#     overall_results.append({
#         "model_type": model_type,
#         "parameter_set": parameter_set,
#         "task": task_label,

#         "alpha": data["alpha"].iloc[0],
#         "K": data["K"].iloc[0],
#         "h_map": data["h_map"].iloc[0],
#         "h_peak": data["h_peak"].iloc[0],
#         "w_mf": data["w_mf"].iloc[0],
#         "w_map": data["w_map"].iloc[0],
#         "eta": data["eta"].iloc[0],

#         "accuracy": data["accuracy"].mean(),
#         "n_trials": data["accuracy"].notna().sum()
#     })

#     # accuracy around changes
#     change_data = extract_change_windows(data = data, task_label = task_label, window = 5)
#     if not change_data.empty:
#         parameter_summary = change_data.groupby(["change_type", "relative_trial"], as_index = False).agg(accuracy=("accuracy", "mean"), n_events=("event_id", "nunique"))
#         parameter_summary["model_type"] = model_type
#         parameter_summary["parameter_set"] = parameter_set
#         parameter_summary["task"] = task_label
#         change_results.append(parameter_summary)


# # save parameter-level results
# overall_results = pd.DataFrame(overall_results)
# overall_results = overall_results.sort_values(["model_type", "task", "parameter_set"]).reset_index(drop = True)
# overall_results.to_csv(os.path.join(output_dir, "overall_accuracy.csv"), index = False)

# change_results = pd.concat(change_results, ignore_index = True)
# change_results = change_results[["model_type", "parameter_set", "task", "change_type", "relative_trial", "accuracy", "n_events"]]
# change_results = change_results.sort_values(["model_type", "task", "parameter_set"]).reset_index(drop = True)
# change_results.to_csv(os.path.join(output_dir,"change_accuracy.csv"), index = False)

# # save model-level results
# overall_model_summary = overall_results.groupby(["model_type", "task"], as_index = False).agg(mean_accuracy = ("accuracy", "mean"), sd_accuracy = ("accuracy", "std"), n_parameter_sets = ("accuracy", "count"))
# overall_model_summary["se_accuracy"] = overall_model_summary["sd_accuracy"] / np.sqrt(overall_model_summary["n_parameter_sets"])
# overall_model_summary.to_csv(os.path.join(output_dir, "overall_accuracy_model_summary.csv"), index = False)

# change_model_summary = change_results.groupby(["model_type", "task", "change_type", "relative_trial"], as_index=False).agg(mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"), n_parameter_sets=("accuracy", "count"))
# change_model_summary["se_accuracy"] = change_model_summary["sd_accuracy"] / np.sqrt(change_model_summary["n_parameter_sets"])
# change_model_summary.to_csv(os.path.join(output_dir, "change_accuracy_model_summary.csv"), index=False)