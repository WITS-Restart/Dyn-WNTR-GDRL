import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Q

from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
#rcParams['axes.grid'] = True

base_folder = "/home/rastafan/Documenti/AssegnoRicerca_2024-25/WNTR-RL"
logs_main_folder = "logs/"


def create_plots_from_logs(subfolder):
    os.chdir(subfolder)

    folder_name = os.path.basename(subfolder)

    num_leaks = re.search(r"leaks(\d+)", folder_name)
    num_leaks = int(num_leaks.group(1)) if num_leaks else 1

    optimal_duration_steps = (2 * num_leaks, 4 * num_leaks)

    # === CONFIG ===
    log_file = "training_log.txt"
    window_size = 10

    pattern = re.compile(
        r"Episode\s+(\d+)/\d+\]\s+Reward:\s+([-+]?\d*\.\d+|\d+),.*?Steps:\s+([-+]?\d*\.\d+|\d+)"
    )

    episodes = []
    rewards = []
    steps = []

    # Parse the log file
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                steps.append(float(match.group(3)))

    # Put data into DataFrame
    df = pd.DataFrame({"Episode": episodes, "Reward": rewards, "Duration": steps})

    # Compute moving averages
    df["Reward Moving Median"] = df["Reward"].rolling(window=window_size).median()
    df["Duration Moving Median"] = df["Duration"].rolling(window=window_size).median()

    # === PLOTS ===
    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(df["Episode"], df["Reward"], alpha=0.3, label="Reward")
    plt.plot(df["Episode"], df["Reward Moving Median"], label=f"Reward Moving Median ({window_size})")
    #plt.hlines([6], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Reward")
    #plt.hlines([4], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Reward")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    #plt.xlim((0, 250))
    plt.title("Reward over Episodes")

    # Duration plot
    plt.subplot(1, 2, 2)
    plt.plot(df["Episode"], df["Duration"], alpha=0.3, label="Duration")
    plt.plot(df["Episode"], df["Duration Moving Median"], label=f"Duration Moving Median ({window_size})")
    optimal_mask = (df["Duration"] >= optimal_duration_steps[0]) & (df["Duration"] <= optimal_duration_steps[1])
    plt.scatter(df.loc[optimal_mask, "Episode"], df.loc[optimal_mask, "Duration"], color="green", s=1, label="Minimal Episode Duration", zorder=5)
    
    #plt.hlines([2], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Duration")
    #plt.hlines([4], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Duration")
    plt.xlabel("Episode")
    plt.ylabel("Duration (steps)")
    plt.legend(markerscale=6, loc="upper right")
    #plt.xlim((0, 250))
    plt.title("Episode Duration in steps")

    plt.tight_layout()

    name = f"../plots/{folder_name}_reward_duration.png"
    plt.savefig(name, dpi=300)

    # ---------- Load ----------
    loss_df = pd.read_csv("losses_dqn.csv", header=None, names=["step", "loss"])
    td_df   = pd.read_csv("td_errors_dqn.csv", header=None, names=["step", "td_error"])
    q_df    = pd.read_csv("q_values_dqn.csv", header=None, names=["step", "q_mean", "q_std"])
    grad_df = pd.read_csv("grad_norm_dqn.csv", header=None, names=["step", "grad_norm"])

    loss_df["loss Moving Median"] = loss_df["loss"].rolling(window=window_size).median()
    td_df["td_error Moving Median"] = td_df["td_error"].rolling(window=window_size).median()
    q_df["q_mean Moving Median"] = q_df["q_mean"].rolling(window=window_size).median()
    q_df["q_std Moving Median"] = q_df["q_std"].rolling(window=window_size).median()
    grad_df["grad_norm Moving Median"] = grad_df["grad_norm"].rolling(window=window_size).median()

    # ---------- Plot ----------
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(loss_df["step"], loss_df["loss"], lw=1)
    axs[0].plot(loss_df["step"], loss_df["loss Moving Median"], color="tab:green", lw=2, label=f"MA({window_size})")
    axs[0].legend()
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss")

    axs[1].plot(loss_df["step"], td_df["td_error"], color="tab:orange", lw=1)
    axs[1].plot(loss_df["step"], td_df["td_error Moving Median"], color="tab:red", lw=2, label=f"MA({window_size})")
    axs[1].legend()
    axs[1].set_ylabel("Count")
    axs[1].set_title("|TD Error| distribution")

    axs[2].plot(q_df["step"], q_df["q_mean"], label="mean Q", lw=1)
    axs[2].plot(q_df["step"], q_df["q_std"], label="std Q", lw=1)
    axs[2].plot(q_df["step"], q_df["q_mean Moving Median"], color="tab:green", lw=2, label=f"mean Q MA({window_size})")
    axs[2].plot(q_df["step"], q_df["q_std Moving Median"], color="tab:red", lw=2, label=f"std Q MA({window_size})")
    axs[2].legend()
    axs[2].set_ylabel("Q-values")
    axs[2].set_title("Q mean / std")

    axs[3].plot(grad_df["step"], grad_df["grad_norm"], color="tab:red", lw=1)
    axs[3].plot(grad_df["step"], grad_df["grad_norm Moving Median"], color="tab:green", lw=2, label=f"MA({window_size})")
    axs[3].legend()
    axs[3].set_ylabel("||grad||₂")
    axs[3].set_xlabel("Training step")
    axs[3].set_title("Gradient norm")

    plt.tight_layout()
    name = f"../plots/{folder_name}_loss_tderr_qmean_grad.png"
    plt.savefig(name, dpi=300)

    plt.close('all')    
    os.chdir(base_folder)


rewards_map = {}
steps_map = {}


def extract_mean_and_std(subfolder):
    os.chdir(subfolder)

    pattern = re.compile(
        r"Episode\s+(\d+)/\d+\]\s+Reward:\s+([-+]?\d*\.\d+|\d+),.*?Steps:\s+([-+]?\d*\.\d+|\d+)"
    )

    episodes = []
    rewards = []
    steps = []

    # Parse the log file
    with open("training_log.txt", "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                steps.append(float(match.group(3)))



    rewards = rewards[900:1000]
    steps = steps[900:1000]

    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5 if rewards else 0

    num_leaks = re.search(r"leaks(\d+)", subfolder)
    num_leaks = int(num_leaks.group(1)) if num_leaks else 1

    #optimal_episodes = [s for s in steps if s <= 4*num_leaks]
    #print(f"Subfolder: {subfolder}, Optimal episodes (steps <=4): {len(optimal_episodes)}/{len(steps)}")

    solved_episodes = [s for s in steps if s < 240]
    print(f"Subfolder: {subfolder}, Solved episodes (steps <240): {len(solved_episodes)}/{len(steps)}")


    mean_steps = sum(steps) / len(steps) if steps else 0
    std_steps = (sum((x - mean_steps) ** 2 for x in steps) / len(steps)) ** 0.5 if steps else 0

    #percentage = int(subfolder.split("_sensors")[-1].split("_demands")[0])
    percentage = subfolder
    rewards_map[percentage] = (mean_reward, std_reward)
    steps_map[percentage] = (mean_steps, std_steps)
    os.chdir(base_folder)





subfolders = [name for name in os.listdir(logs_main_folder) if os.path.isdir(os.path.join(logs_main_folder, name))]

if len(subfolders) == 0:
    raise ValueError("No log folders found in the logs directory.")
elif len(subfolders) == 1:
    log_folder = os.path.join(logs_main_folder, subfolders[0])
    print(f"Only one log folder found. Using: {log_folder}")
else:
    #selected = input("Available log folders:\n" + "\n".join(f"{i}: {name}" for i, name in enumerate(subfolders)) + "\n> ")
    #if not selected or not selected.isdigit() or int(selected) < 0 or int(selected) >= len(subfolders):
    #    raise ValueError("Invalid selection.") 
    #log_folder = os.path.join(logs_main_folder, subfolders[int(selected)])
    #if not os.path.exists(log_folder):
    #    raise ValueError(f"Log folder {log_folder} does not exist.")
    for folder in subfolders:
        #folders = [
        #"Training_WDN_DQN_20x20_branched_sensors100_demands5_leaks1", 
        #"Training_WDN_DQN_20x20_branched_sensors95_demands5_leaks1", 
        #"Training_WDN_DQN_20x20_branched_sensors90_demands5_leaks1", 
        #"Training_WDN_DQN_20x20_branched_sensors80_demands5_leaks1", 
        #"Training_WDN_DQN_20x20_branched_sensors70_demands5_leaks1", 
        #"Training_WDN_DQN_20x20_branched_sensors60_demands5_leaks1", 
        #"Training_WDN_DQN_20x20_branched_sensors50_demands5_leaks1"]
        
        #if folder not in folders:
        if "100" not in folder or "bak" in folder:
            #print(f"Skipping folder: {folder}")
            continue
        log_folder = os.path.join(logs_main_folder, folder)
        print(f"Processing log folder: {log_folder}")
        try:
            extract_mean_and_std(log_folder)
            #create_plots_from_logs(log_folder)
        except Exception as e:
            print(f"Error processing folder {log_folder}: {e}")

    if len(rewards_map) == 0:
        raise ValueError("No valid log data found in the specified folders.")
    # plot mean and std with x as percentage of sensors but in reverse from 100 to 50

    print("Rewards map:", rewards_map)
    print("Steps map:", steps_map)
    exit(0)

    percentages = sorted(rewards_map.keys(), reverse=True)
    means_rew = [rewards_map[p][0] for p in percentages]
    stds_rew = [rewards_map[p][1] for p in percentages]

    means_steps = [steps_map[p][0] for p in percentages]
    stds_steps = [steps_map[p][1] for p in percentages]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_rew = "tab:blue"
    color_steps = "tab:orange"

    ln1 = ax1.errorbar(percentages, means_rew, yerr=stds_rew, fmt='-o', capsize=5, color=color_rew, label="Mean Reward")
    ax1.set_xlabel("Percentage of Sensors")
    ax1.set_ylabel("Mean Reward", color=color_rew)
    ax1.tick_params(axis="y", labelcolor=color_rew)
    ax1.set_xticks(percentages)
    ax1.grid(True)

    ax2 = ax1.twinx()
    min_allowed = 2
    max_allowed = 240
    means_steps_arr = np.array(means_steps, dtype=float)
    stds_steps_arr = np.array(stds_steps, dtype=float)
    lower_err = np.clip(np.minimum(stds_steps_arr, means_steps_arr - min_allowed), 0.0, None)
    upper_err = np.clip(np.minimum(stds_steps_arr, max_allowed - means_steps_arr), 0.0, None)

    ln2 = ax2.errorbar(percentages, means_steps, yerr=[lower_err, upper_err], fmt='-s', capsize=5, color=color_steps, label="Mean Episode Duration (steps)")
    #ln2 = ax2.errorbar(percentages, means_steps, yerr=stds_steps, fmt='-s', capsize=5, color=color_steps, label="Mean Episode Duration (steps)")
    ax2.set_ylabel("Mean Episode Duration (steps)", color=color_steps)
    ax2.tick_params(axis="y", labelcolor=color_steps)
    #ax2.grid(True)

    # combined legend
    handles = []
    labels = []
    for ln in (ln1, ln2):
        if isinstance(ln, tuple) or hasattr(ln, "legend_elements"):
            # errorbar returns a container; get the Line2D object(s)
            line = ln[0] if isinstance(ln, tuple) else ln
        else:
            line = ln
        handles.append(line)
        labels.append("Mean Reward" if ln == ln1 else "Mean Episode Duration (steps)")

    ax1.legend(handles, labels, loc="best")

    plt.tight_layout()
    plt.savefig("logs/plots/mean_reward_vs_sensors.png", dpi=300)
    plt.close(fig)
    
    from sys import exit
    exit(0)

os.chdir(log_folder)

# === CONFIG ===
log_file = "training_log.txt"
window_size = 10

pattern = re.compile(
    r"Episode\s+(\d+)/\d+\]\s+Reward:\s+([-+]?\d*\.\d+|\d+),.*?Steps:\s+([-+]?\d*\.\d+|\d+)"
)

episodes = []
rewards_map = []
steps_map = []

# Parse the log file
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            episodes.append(int(match.group(1)))
            rewards_map.append(float(match.group(2)))
            steps_map.append(float(match.group(3)))

# Put data into DataFrame
df = pd.DataFrame({"Episode": episodes, "Reward": rewards_map, "Duration": steps_map})

# Compute moving averages
df["Reward_MA"] = df["Reward"].rolling(window=window_size).median()
df["Duration_MA"] = df["Duration"].rolling(window=window_size).median()

# === PLOTS ===
plt.figure(figsize=(12, 5))

# Reward plot
plt.subplot(1, 2, 1)
plt.plot(df["Episode"], df["Reward"], alpha=0.3, label="Reward")
plt.plot(df["Episode"], df["Reward_MA"], label=f"Reward MA ({window_size})")
plt.hlines([6], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Reward")
plt.hlines([4], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
#plt.xlim((0, 250))
plt.title("Reward over Episodes")

# Duration plot
plt.subplot(1, 2, 2)
plt.plot(df["Episode"], df["Duration"], alpha=0.3, label="Duration")
plt.plot(df["Episode"], df["Duration_MA"], label=f"Duration MA ({window_size})")
plt.hlines([2], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Duration")
plt.hlines([4], xmin=df["Episode"].min(), xmax=df["Episode"].max(), colors='r', linestyles='dashed', label="Optimal Duration")
plt.xlabel("Episode")
plt.ylabel("Duration (steps)")
plt.legend()
#plt.xlim((0, 250))
plt.title("Episode Duration in steps")

plt.tight_layout()
plt.show(block = False)

# ---------- Load ----------
loss_df = pd.read_csv("losses_dqn.csv", header=None, names=["step", "loss"])
td_df   = pd.read_csv("td_errors_dqn.csv", header=None, names=["step", "td_error"])
q_df    = pd.read_csv("q_values_dqn.csv", header=None, names=["step", "q_mean", "q_std"])
grad_df = pd.read_csv("grad_norm_dqn.csv", header=None, names=["step", "grad_norm"])

loss_df["loss_MA"] = loss_df["loss"].rolling(window=window_size).median()
td_df["td_error_MA"] = td_df["td_error"].rolling(window=window_size).median()
q_df["q_mean_MA"] = q_df["q_mean"].rolling(window=window_size).median()
q_df["q_std_MA"] = q_df["q_std"].rolling(window=window_size).median()
grad_df["grad_norm_MA"] = grad_df["grad_norm"].rolling(window=window_size).median()

# ---------- Plot ----------
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axs[0].plot(loss_df["step"], loss_df["loss"], lw=1)
axs[0].plot(loss_df["step"], loss_df["loss_MA"], color="tab:green", lw=2, label=f"MA({window_size})")
axs[0].legend()
axs[0].set_ylabel("Loss")
axs[0].set_title("Training Loss")

axs[1].plot(loss_df["step"], td_df["td_error"], color="tab:orange", lw=1)
axs[1].plot(loss_df["step"], td_df["td_error_MA"], color="tab:red", lw=2, label=f"MA({window_size})")
axs[1].legend()
axs[1].set_ylabel("Count")
axs[1].set_title("|TD Error| distribution")

axs[2].plot(q_df["step"], q_df["q_mean"], label="mean Q", lw=1)
axs[2].plot(q_df["step"], q_df["q_std"], label="std Q", lw=1)
axs[2].plot(q_df["step"], q_df["q_mean_MA"], color="tab:green", lw=2, label=f"mean Q MA({window_size})")
axs[2].plot(q_df["step"], q_df["q_std_MA"], color="tab:red", lw=2, label=f"std Q MA({window_size})")
axs[2].legend()
axs[2].set_ylabel("Q-values")
axs[2].set_title("Q mean / std")

axs[3].plot(grad_df["step"], grad_df["grad_norm"], color="tab:red", lw=1)
axs[3].plot(grad_df["step"], grad_df["grad_norm_MA"], color="tab:green", lw=2, label=f"MA({window_size})")
axs[3].legend()
axs[3].set_ylabel("||grad||₂")
axs[3].set_xlabel("Training step")
axs[3].set_title("Gradient norm")

plt.tight_layout()
plt.show()
