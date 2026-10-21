#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# get current path
script_dir = os.path.dirname(os.path.abspath(__file__))
human_dir = os.path.dirname(script_dir)

model_overall_accuracy = pd.read_csv(os.path.join(script_dir, "overall_accuracy.csv"))
model_change_accuracy = pd.read_csv(os.path.join(script_dir, "change_accuracy.csv"))

human_data_D3 = pd.read_csv(os.path.join(human_dir, "human_results/expdata_D3.csv"))
human_data_D4 = pd.read_csv(os.path.join(human_dir, "human_results/expdata_D4.csv"))
human_data_D3 = human_data_D3[human_data_D3["task"].isin(["choice task single ring_one", "choice task single ring_two"])]
human_data_D3 = human_data_D3[["subj_id","day","task","blockID","trialID","context","ring","ring_label","reward_dist","left_choice","right_choice","left_reward","right_reward","choice","reward"]]
human_data_D4 = human_data_D4[human_data_D4["task"] == "choice task two context"]
human_data_D4 = human_data_D4[["subj_id","day","task","blockID","trialID","context","ring","ring_label","reward_dist","left_choice","right_choice","left_reward","right_reward","choice","reward"]]
human_data = pd.concat([human_data_D3, human_data_D4], ignore_index = True)

# %%
# preprocessing human data
# compute accuracy
def compute_accuracy(data):
    data = data.copy()

    data["chosen_reward"] = np.where(
        data["choice"] == 'left',
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

        previous_peak = block["reward_dist"].shift(1)
        previous_map = block["ring"].shift(1)

        peak_change = ((block["reward_dist"] != previous_peak) & previous_peak.notna())

        if task_label in ["single_map_one", "single_map_two"]:
            events = [(position, "peak_change") for position in np.where(peak_change)[0]]
        else:
            context_change = ((block["ring"] != previous_map) & previous_map.notna())
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

human_overall_accuracy = []
human_change_accuracy = []

for subj_id in human_data["subj_id"].unique():
    data = human_data[human_data["subj_id"] == subj_id]
    data = compute_accuracy(data)

    # compute accuracy in three choice tasks
    for task in data["task"].unique():
        task_data = data[data["task"] == task]
        if task == "choice task single ring_one":
            task_label = "single_map_one"
        elif task == "choice task single ring_two":
            task_label = "single_map_two"
        else:
            task_label = "double_map"

        human_overall_accuracy.append({
            "subj_id": subj_id,
            "task": task_label,
            "accuracy": task_data["accuracy"].mean(),
            "n_trials": task_data["accuracy"].notna().sum()
        })

        # accuracy around changes
        change_data = extract_change_windows(data = task_data, task_label = task_label, window = 5)
        if not change_data.empty:
            subject_summary = change_data.groupby(["change_type", "relative_trial"], as_index = False).agg(accuracy = ("accuracy", "mean"), n_events = ("event_id", "nunique"))
            subject_summary["subj_id"] = subj_id
            subject_summary["task"] = task_label
            human_change_accuracy.append(subject_summary)

human_overall_accuracy = pd.DataFrame(human_overall_accuracy)
human_overall_accuracy = human_overall_accuracy.sort_values(["subj_id", "task"]).reset_index(drop = True)
human_change_accuracy = pd.concat(human_change_accuracy, ignore_index=True)
human_change_accuracy = human_change_accuracy[["subj_id", "task", "change_type", "relative_trial", "accuracy", "n_events"]]
human_change_accuracy = human_change_accuracy.sort_values(["subj_id", "task"]).reset_index(drop = True)


# %%
# ploting settings
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12
}) 

# plotting accuracy
model_accuracy_data = model_overall_accuracy.copy()
model_accuracy_data['group'] = model_accuracy_data['model_type']
model_accuracy_data = model_accuracy_data.rename(columns = {"parameter_set": "subj_id"})
model_accuracy_data = model_accuracy_data[['subj_id', 'group', 'task', 'accuracy', 'n_trials']]
human_overall_accuracy['group'] = 'human'
accuracy_plot_data = pd.concat([model_accuracy_data, human_overall_accuracy], ignore_index = True)

fig, ax = plt.subplots(figsize = (20, 5))
sns.boxplot(data = accuracy_plot_data, x = "group", y= "accuracy", hue = 'task', ax = ax)
ax.axhline(y = 0.5, color = "grey", linestyle = "--", linewidth = 1.5)
ax.set_yticks(np.arange(0.4, 1.1, 0.1))
ax.set_xlabel("")
ax.set_ylabel("accuracy")
ax.set_title("Overall Accuracy", fontsize = 18, fontweight = "bold")
ax.legend(title = None)
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(script_dir, "plot/overall_accuracy_human_model.png"), dpi = 300, bbox_inches = "tight")

#%%
# plotting trial-wise accuracy, peak
model_change_data = model_change_accuracy.copy()
model_change_data['group'] = model_change_data['model_type']
model_change_data = model_change_data.rename(columns = {"parameter_set": "subj_id"})
model_change_data = model_change_data[['subj_id','group','task','change_type','relative_trial','accuracy', 'n_events']]
human_change_accuracy['group'] = 'human'
change_plot_data = pd.concat([model_change_data, human_change_accuracy], ignore_index = True)

all_xticks = []
all_xticklabels = []

model_order = [
    "flat_Decay",
    "flat_NoDecay",
    "hybrid_FreeWeights_Decay",
    "hybrid_FreeWeights_NoDecay",
    "hybrid_Pearce_Hall_Decay",
    "hybrid_Pearce_Hall_NoDecay",
    "human"
]

change_plot_summary = change_plot_data.groupby(["group", "relative_trial", "change_type"], as_index = False).agg(
        mean_accuracy=("accuracy", "mean"), se_accuracy=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))))
trial_order = sorted(change_plot_summary["relative_trial"].unique())
n_trials = len(trial_order)
gap = 2

xtick_positions = []
xtick_labels = []

# change = 'peak_only_change'
# change = 'context_change'
for change in ['peak_only_change', 'context_change']:
    fig, ax = plt.subplots(figsize = (20, 5))
    for i, model_name in enumerate(model_order):
        change_data = change_plot_summary[change_plot_summary["group"] == model_name].copy()
        change_data = change_data[change_data['change_type'] == change]
        change_data = change_data.sort_values("relative_trial")

        x = np.arange(n_trials) + i * (n_trials + gap)
        y = change_data["mean_accuracy"].to_numpy()
        se = change_data["se_accuracy"].to_numpy()

        ax.plot(x, y, linewidth=2)
        ax.fill_between(x, y - se, y + se, alpha=0.4)

        all_xticks.extend(x)
        all_xticklabels.extend(change_data["relative_trial"].tolist())

        # model label 
        ax.text(x.mean(), -0.1, model_name,ha="center", va="top",
            transform=ax.get_xaxis_transform(),fontsize = 13)

        # add change point line
        zero_idx = change_data.index[change_data["relative_trial"] == 0][0]
        zero_pos = x[list(change_data["relative_trial"]).index(0)]
        ax.axvline(zero_pos, color="grey", linestyle="--", linewidth=1)

    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xticklabels)
    ax.set_title(f"Trial-wise Accuracy, {change}", fontsize = 18, fontweight = "bold")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.axhline(y = 0.5, color="grey", linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(script_dir, f"plot/trialwise_accuracy_{change}_human_model.png"), dpi = 300, bbox_inches = "tight")

# %%
