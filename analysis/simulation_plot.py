#%%
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# get current path
script_dir = os.path.dirname(os.path.abspath(__file__))
human_dir = os.path.dirname(script_dir)

# model_overall_accuracy = pd.read_csv(os.path.join(script_dir, "results_6models/overall_accuracy.csv"))
# model_change_accuracy = pd.read_csv(os.path.join(script_dir, "results_6models/change_accuracy.csv"))
model_overall_accuracy = pd.read_csv(os.path.join(script_dir, "results_7models/overall_accuracy.csv"))
model_change_accuracy = pd.read_csv(os.path.join(script_dir, "results_7models/change_accuracy.csv"))

human_data_D3 = pd.read_csv(os.path.join(human_dir, "human_results/expdata_D3.csv"))
human_data_D4 = pd.read_csv(os.path.join(human_dir, "human_results/expdata_D4.csv"))
human_data_D3 = human_data_D3[human_data_D3["task"].isin(["choice task single ring_one", "choice task single ring_two"])]
human_data_D3 = human_data_D3[["subj_id","day","task","blockID","trialID","context","ring","ring_label","reward_dist","left_choice","right_choice","left_reward","right_reward","choice","reward"]]
human_data_D4 = human_data_D4[human_data_D4["task"] == "choice task two context"]
human_data_D4 = human_data_D4[["subj_id","day","task","blockID","trialID","context","ring","ring_label","reward_dist","left_choice","right_choice","left_reward","right_reward","choice","reward"]]
human_data = pd.concat([human_data_D3, human_data_D4], ignore_index = True)

human_relevance_model_results_D3 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/relevance_model_results_D3.csv"))
human_relevance_model_results_D4 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/relevance_model_results_D4.csv"))
human_identity_model_results_D3 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/identity_model_results_D3.csv"))
human_identity_model_results_D4 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/identity_model_results_D4.csv"))
human_peak_change_model_results_D3 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/peak_change_beta_D3.csv"))
human_peak_change_model_results_D4 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/peak_change_beta_D4.csv"))
human_context_change_model_results_D4 = pd.read_csv(os.path.join(human_dir, "human_results/human_regression_results/context_change_beta_D4.csv"))

human_regression_dir = os.path.join(human_dir, "human_results/human_regression_results")
model_regression_dir = os.path.join(script_dir, "model_regression_results")

models = ["flat_NoDecay", "flat_Decay", "hybrid_FreeWeights_NoDecay", 
          "hybrid_FreeWeights_Decay", "hybrid_Pearce_Hall_NoDecay", 
          "hybrid_Pearce_Hall_Decay", "full_Bayesian"]

regression_labels = {
    "identity": ["D3_identity", "D4_identity"],
    "relevance": ["D3_relevance", "D4_relevance"],
    "peak_change": ["D3_peak_change", "D4_peak_change"],
    "context_change": ["D4_context_change"],
}

human_files = {
    "identity": {
        "D3_identity": "identity_model_results_D3.csv",
        "D4_identity": "identity_model_results_D4.csv"
    },
    "relevance": {
        "D3_relevance": "relevance_model_results_D3.csv",
        "D4_relevance": "relevance_model_results_D4.csv"
    },
    "peak_change": {
        "D3_peak_change": "peak_change_beta_D3.csv",
        "D4_peak_change": "peak_change_beta_D4.csv"
    },
    "context_change": {
        "D4_context_change": "context_change_beta_D4.csv"
    }
}

regression_results = {name: [] for name in regression_labels}

# loading human results
for category, file_map in human_files.items():
    for regression, filename in file_map.items():

        df = pd.read_csv(os.path.join(human_regression_dir, filename))

        if category in ["identity", "relevance"]:
            df = df.loc[df["family"] == "above_zero"].copy()
            df = df.rename(columns={"comparison": "term"})
        else:
            df = df.rename(columns={
                "predictor": "term",
                "conf_low": "ci_low",
                "conf_high": "ci_high"
            })

        df["source"] = "human"
        df["day"] = regression.split("_")[0]

        regression_results[category].append(df)


# loading simulation results
for model in models:

    model_dir = os.path.join(model_regression_dir, model, "bayesian")

    group_beta = pd.read_csv(os.path.join(model_dir, "group_beta.csv"))
    position_beta = pd.read_csv(os.path.join(model_dir, "position_beta.csv"))

    for category, labels in regression_labels.items():

        df = group_beta if category in ["identity", "relevance"] else position_beta

        subset = df.loc[df["regression"].isin(labels)].copy()

        subset = subset.rename(columns={
            "predictor": "term",
            "conf_low": "ci_low",
            "conf_high": "ci_high"
            # "hdi_2.5%": "ci_low",
            # "hdi_97.5%": "ci_high",
            # "hdi_2.5": "ci_low",
            # "hdi_97.5": "ci_high"
        })

        subset["source"] = model
        subset["day"] = subset["regression"].str.split("_").str[0]

        regression_results[category].append(subset)


# combine and sorting columns
def combine_results(category):
    df = pd.concat(regression_results[category], ignore_index=True)

    columns = ["source", "day", "term", "estimate", "ci_low", "ci_high"]

    if category in ["peak_change", "context_change"]:
        columns.insert(2, "position")

    df = df.loc[
        ~df["term"].isin(["Intercept", "(Intercept)"]),
        columns
    ].reset_index(drop=True)

    return df


identity_results = combine_results("identity")
relevance_results = combine_results("relevance")
peak_change_results = combine_results("peak_change")
context_change_results = combine_results("context_change")

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
    "font.size": 24,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
}) 

# plotting accuracy
model_accuracy_data = model_overall_accuracy.copy()
model_accuracy_data['group'] = model_accuracy_data['model_type']
model_accuracy_data = model_accuracy_data.rename(columns = {"parameter_set": "subj_id"})
model_accuracy_data = model_accuracy_data[['subj_id', 'group', 'task', 'accuracy', 'n_trials']]
human_overall_accuracy['group'] = 'human'
accuracy_plot_data = pd.concat([model_accuracy_data, human_overall_accuracy], ignore_index = True)

fig, ax = plt.subplots(figsize = (32, 6))
sns.boxplot(data = accuracy_plot_data, x = "group", y= "accuracy", hue = 'task', ax = ax)
ax.axhline(y = 0.5, color = "grey", linestyle = "--", linewidth = 1.5)
ax.set_yticks(np.arange(0.4, 1.1, 0.1))
ax.set_xlabel("")
ax.set_ylabel("accuracy")
ax.set_title("Overall Accuracy", fontsize = 36, fontweight = "bold")
ax.legend(title = None)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
# plt.show()

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

# model_order = ["flat_NoDecay", "flat_Decay", "hybrid_FreeWeights_NoDecay",
#                "hybrid_FreeWeights_Decay", "hybrid_Pearce_Hall_NoDecay",
#                "hybrid_Pearce_Hall_Decay"]

model_order = ["flat_NoDecay", "flat_Decay", "hybrid_FreeWeights_NoDecay",
               "hybrid_FreeWeights_Decay", "hybrid_Pearce_Hall_NoDecay",
               "hybrid_Pearce_Hall_Decay", "full_Bayesian", "human"]

change_plot_summary = change_plot_data.groupby(["group", "relative_trial", "change_type"], as_index = False).agg(
        mean_accuracy = ("accuracy", "mean"), se_accuracy=("accuracy", lambda x: x.std(ddof = 1) / np.sqrt(len(x))))
trial_order = sorted(change_plot_summary["relative_trial"].unique())
n_trials = len(trial_order)
gap = 2

xtick_positions = []
xtick_labels = []

for change in ['peak_only_change', 'context_change', 'peak_change']:
    fig, ax = plt.subplots(figsize = (32, 6))
    for i, model_name in enumerate(model_order):
        change_data = change_plot_summary[change_plot_summary["group"] == model_name].copy()
        change_data = change_data[change_data['change_type'] == change]
        change_data = change_data.sort_values("relative_trial")

        x = np.arange(n_trials) + i * (n_trials + gap)
        y = change_data["mean_accuracy"].to_numpy()
        se = change_data["se_accuracy"].to_numpy()

        ax.plot(x, y, linewidth = 2)
        ax.fill_between(x, y - se, y + se, alpha = 0.4)

        all_xticks.extend(x)
        all_xticklabels.extend(change_data["relative_trial"].tolist())

        # model label 
        ax.text(x.mean(), -0.1, model_name, ha = "center", va = "top",
            transform = ax.get_xaxis_transform())

        # add change point line
        zero_idx = change_data.index[change_data["relative_trial"] == 0][0]
        zero_pos = x[list(change_data["relative_trial"]).index(0)]
        ax.axvline(zero_pos, color = "grey", linestyle = "--", linewidth = 1)

    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xticklabels)
    ax.set_title(f"Trial-wise Accuracy, {change}", fontsize = 36, fontweight = "bold")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.axhline(y = 0.5, color = "grey", linestyle = "--", linewidth = 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    # plt.show()

    fig.savefig(os.path.join(script_dir, f"plot/trialwise_accuracy_{change}_human_model.png"), dpi = 300, bbox_inches = "tight")

# plotting regression results
plot_order = ["flat_NoDecay", "flat_Decay", "hybrid_FreeWeights_NoDecay",
              "hybrid_FreeWeights_Decay", "hybrid_Pearce_Hall_NoDecay",
              "hybrid_Pearce_Hall_Decay", "full_Bayesian", "human"]

plot_data = {
    "identity": identity_results,
    "relevance": relevance_results,
    "peak_change": peak_change_results,
    "context_change": context_change_results
}

term_order = {
    "identity": ["experience_diff", "R1_diff", "R2_diff"],
    "relevance": ["experience_diff", "rel_diff", "irrel_diff"],
    "peak_change": ["experience_diff", "strategy1_diff", "strategy2_diff",
                    "strategy3_diff", "strategy4_diff"],
    "context_change": ["experience_diff", "strategy1_diff", "strategy2_diff",
                       "strategy3_diff", "strategy4_diff"]
}

# relevance model results
for day in sorted(relevance_results["day"].unique()):
    plot_df = relevance_results[relevance_results["day"] == day].copy()
    terms = term_order['relevance']
    gap = 2
    all_xticks, all_xticklabels = [], []
    fig, ax = plt.subplots(figsize = (32, 6))

    for i, model_name in enumerate(plot_order):
        model_data = plot_df[plot_df["source"] == model_name].copy()
        offset = i * (len(terms) + gap)
        x_all = np.arange(len(terms)) + offset
        
        for j, term in enumerate(terms):
            term_data = model_data[model_data["term"] == term]
            row = term_data.iloc[0]
            ax.errorbar(x_all[j], row["estimate"], yerr = [[row["estimate"] - row["ci_low"]],
                        [row["ci_high"] - row["estimate"]]], fmt = "o", color = f"C{j}", 
                        capsize = 4, markersize = 7)

        all_xticks.extend(x_all)
        all_xticklabels.extend(terms)
        y_pos = -0.35 if i % 2 == 0 else -0.45
        ax.text(x_all.mean(), y_pos, model_name, ha = "center", va = "top",
                transform = ax.get_xaxis_transform())
        ax.set_xticks(all_xticks)
        ax.set_xticklabels(all_xticklabels, rotation = 60, ha = "right", rotation_mode = "anchor")
    ax.set_ylim(-1, 4)
    ax.set_yticks(np.arange(-1, 4, 0.5))
    ax.axhline(0, color = "grey", linestyle = "--", linewidth = 1)
    ax.set_ylabel("Beta")
    ax.set_title(f"Relevance regression model, {day}", fontsize = 36, fontweight = "bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon = False, loc = "upper right")

    # plt.tight_layout()
    # plt.show()

    fig.savefig(os.path.join(script_dir, f"plot/relevance_{day}_human_model.png"), dpi = 300, bbox_inches = "tight")

# identity model results
for day in sorted(identity_results["day"].unique()):
    plot_df = identity_results[identity_results["day"] == day].copy()
    terms = term_order['identity']
    gap = 2
    all_xticks, all_xticklabels = [], []
    fig, ax = plt.subplots(figsize = (32, 6))

    for i, model_name in enumerate(plot_order):
        model_data = plot_df[plot_df["source"] == model_name].copy()
        offset = i * (len(terms) + gap)
        x_all = np.arange(len(terms)) + offset
        
        for j, term in enumerate(terms):
            term_data = model_data[model_data["term"] == term]
            row = term_data.iloc[0]
            ax.errorbar(x_all[j], row["estimate"], yerr = [[row["estimate"] - row["ci_low"]],
                        [row["ci_high"] - row["estimate"]]], fmt = "o", color = f"C{j}", 
                        capsize = 4, markersize = 7)

        all_xticks.extend(x_all)
        all_xticklabels.extend(terms)
        y_pos = -0.35 if i % 2 == 0 else -0.45
        ax.text(x_all.mean(), y_pos, model_name, ha = "center", va = "top",
                transform = ax.get_xaxis_transform())
        ax.set_xticks(all_xticks)
        ax.set_xticklabels(all_xticklabels, rotation = 60, ha = "center")
    ax.set_ylim(-1, 4)
    ax.set_yticks(np.arange(-1, 4, 0.5))
    ax.axhline(0, color = "grey", linestyle = "--", linewidth = 1)
    ax.set_ylabel("Beta")
    ax.set_title(f"Identity regression model, {day}", fontsize = 36, fontweight = "bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon = False, loc = "upper right")

    # plt.tight_layout()
    # plt.show()

    fig.savefig(os.path.join(script_dir, f"plot/identity_{day}_human_model.png"), dpi = 300, bbox_inches = "tight")

# context change results
terms = term_order['context_change']
gap = 2
all_xticks, all_xticklabels = [], []
fig, ax = plt.subplots(figsize = (32, 6))
positions = np.sort(context_change_results["position"].dropna().unique())
n_positions = len(positions)

for i, model_name in enumerate(plot_order):
    model_data = context_change_results[context_change_results["source"] == model_name].copy()
    offset = i * (n_positions + gap)
    x_all = np.arange(n_positions) + offset

    for j, term in enumerate(terms):
        term_data = model_data[model_data["term"] == term].sort_values("position")

        x = np.searchsorted(positions, term_data["position"]) + offset
        y = term_data["estimate"].to_numpy()
        low = term_data["ci_low"].to_numpy()
        high = term_data["ci_high"].to_numpy()

        ax.plot(x, y, linewidth = 2, color = f"C{j}",
                label = term if i == 0 else None)
        ax.fill_between(x, low, high, color = f"C{j}", alpha = 0.15)

    if 0 in positions:
        ax.axvline(offset + np.where(positions == 0)[0][0], color = "grey", 
                    linestyle = "--", linewidth = 1)

    all_xticks.extend(x_all)
    all_xticklabels.extend(positions)
    y_pos = -0.1 if i % 2 == 0 else -0.2
    ax.text(x_all.mean(), y_pos, model_name, ha = "center", va = "top",
            transform = ax.get_xaxis_transform())
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xticklabels, ha = "center")
ax.set_ylim(-1, 4)
ax.set_yticks(np.arange(-1, 4, 0.5))
ax.axhline(0, color = "grey", linestyle = "--", linewidth = 1)
ax.set_ylabel("Beta")
ax.set_title(f"Context change regression model, {day}", fontsize = 36, fontweight = "bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon = False, loc = "upper right")

plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(script_dir, f"plot/context_change_{day}_human_model.png"), dpi = 300, bbox_inches = "tight")

# peak change results
for day in sorted(peak_change_results["day"].unique()):
    plot_df = peak_change_results[peak_change_results["day"] == day].copy()
    terms = term_order['peak_change']
    gap = 2
    all_xticks, all_xticklabels = [], []
    fig, ax = plt.subplots(figsize = (32, 6))
    positions = np.sort(plot_df["position"].dropna().unique())
    n_positions = len(positions)

    for i, model_name in enumerate(plot_order):
        model_data = plot_df[plot_df["source"] == model_name].copy()
        offset = i * (n_positions + gap)
        x_all = np.arange(n_positions) + offset

        for j, term in enumerate(terms):
            term_data = model_data[model_data["term"] == term].sort_values("position")

            x = np.searchsorted(positions, term_data["position"]) + offset
            y = term_data["estimate"].to_numpy()
            low = term_data["ci_low"].to_numpy()
            high = term_data["ci_high"].to_numpy()

            ax.plot(x, y, linewidth = 2, color = f"C{j}",
                    label = term if i == 0 else None)
            ax.fill_between(x, low, high, color = f"C{j}", alpha = 0.15)

        if 0 in positions:
            ax.axvline(offset + np.where(positions == 0)[0][0], color = "grey", 
                       linestyle = "--", linewidth = 1)

        all_xticks.extend(x_all)
        all_xticklabels.extend(positions)
        y_pos = -0.1 if i % 2 == 0 else -0.2
        ax.text(x_all.mean(), y_pos, model_name, ha = "center", va = "top",
                transform = ax.get_xaxis_transform())
        ax.set_xticks(all_xticks)
        ax.set_xticklabels(all_xticklabels, ha = "center")
    ax.set_ylim(-1, 4)
    ax.set_yticks(np.arange(-1, 4, 0.5))
    ax.axhline(0, color = "grey", linestyle = "--", linewidth = 1)
    ax.set_ylabel("Beta")
    ax.set_title(f"Peak change regression model, {day}", fontsize = 36, fontweight = "bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon = False, loc = "upper right")

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(script_dir, f"plot/peak_change_{day}_human_model.png"), dpi = 300, bbox_inches = "tight")

