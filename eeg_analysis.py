import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm

from moabb.analysis.meta_analysis import compute_dataset_statistics, compute_dataset_statistics, find_significant_differences

# --- Load data ---
save_path = "./results_eeg"
classes = ["ft", "lh", "rh", "tn"]
classes_string = "-".join(classes)

# Set path, datasets and classes to analyze
results_paths = [save_path]

pipeline_names = [
            "COV+MDM", "COV+TSP+LR", # Standard Riemannian pipelines
            "ACOV+MDM", "ACOV+TSH+LR", # ACOV-based Riemannian pipelines
            ]


# ===================================================
# ===== Average accuracy and standard deviation =====
# ===================================================

# -- For each classifier caclulate average result and standard deviation ---
for results_path in results_paths:

    print("Average accuracy and standard deviation for results in", results_path)
    
    for clf_name in pipeline_names:
        file_path = os.path.join(results_path, "results_{}.csv".format(clf_name))
        
        # Check if the file exists before processing
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            score_column = df["score"]
            print("{}: \t".format(clf_name), np.mean(score_column), np.std(score_column))
        else:
            print("{}: \t File not found for classifier".format(clf_name))
print()

# =========================================
# ===== Statistical tests using MOABB =====
# =========================================

# --- Step 1: read the all results of all pipelines we want to test into one big dataframe  ---

results_all = pd.DataFrame()

for clf_name in pipeline_names: 

    results_clf_name = pd.DataFrame()
    file_path = os.path.join(results_path, "results_{}.csv".format(clf_name))
    
    # Check if the file exists before processing
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        score_column = df["score"]
        if score_column.empty:
            print("Empty score column for {} in {}".format(clf_name, results_path))
            continue
        else:
            results_all = pd.concat([results_all, df], ignore_index=True)
    else:
        print("File not found for {} in {}".format(clf_name, results_path))
        continue            

# Store results per dataset as well
results_per_dataset = {}

print("All results combined:")
print(results_all, "\n")
#results_all["dataset"].unique(), results_all.describe()


# --- Helper function to plot and save significant differences ---
def plot_save_significant_differences(sig_pval, T, title_suffix="", save_path=None):
    """
    Plot and optionally save the significant differences matrix and T-values matrix.

    Parameters:
    - sig_pval: DataFrame or 2D array of p-values between classifiers.
    - T: DataFrame or 2D array of T-values between classifiers.
    - title_suffix: Optional string to append to the plot titles.
    - save_path: Optional path to save the plots. If None, plots are not saved.
    """

    # --- T-value plot (masked, only show where p < 0.05) ---

    # Mask T-values where p-value >= 0.05
    T_masked = T.where(sig_pval < 0.05)

    data_T = T_masked.values if hasattr(T_masked, "values") else T_masked
    labels_T = T_masked.index if hasattr(T_masked, "index") else np.arange(data_T.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_T, cmap="viridis", vmin=np.nanmin(data_T), vmax=np.nanmax(data_T))

    # Grid fix:
    ax.grid(False)
    # Set minor ticks at the edges of each cell
    ax.set_xticks(np.arange(data_T.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(data_T.shape[0]+1)-0.5, minor=True)
    # Draw gridlines at minor ticks
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data_T.shape[0]):
        for j in range(data_T.shape[1]):
            if not np.isnan(data_T[i, j]):
                ax.text(j, i, f"{data_T[i, j]:.2f}", ha="center", va="center", color="black", fontsize=12)

    ax.set_xticks(np.arange(len(labels_T)))
    ax.set_yticks(np.arange(len(labels_T)))
    ax.set_xticklabels(labels_T, rotation=45, ha='right')
    ax.set_yticklabels(labels_T)

    plt.title("T values (only where p < 0.05)")
    plt.colorbar(im, ax=ax, label="T value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"t_values_{title_suffix}.png")) if save_path else None


    # --- p-value plot (show all p-values) ---

    data_p = sig_pval.values if hasattr(sig_pval, "values") else sig_pval
    labels_p = sig_pval.index if hasattr(sig_pval, "index") else np.arange(data_p.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_p, cmap="viridis", vmin=np.nanmin(data_p), vmax=np.nanmax(data_p))

    # Grid fix:
    ax.grid(False)
    # Set minor ticks at the edges of each cell
    ax.set_xticks(np.arange(data_T.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(data_T.shape[0]+1)-0.5, minor=True)
    # Draw gridlines at minor ticks
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data_p.shape[0]):
        for j in range(data_p.shape[1]):
            ax.text(j, i, f"{data_p[i, j]:.2g}", ha="center", va="center", color="black", fontsize=12)

    ax.set_xticks(np.arange(len(labels_p)))
    ax.set_yticks(np.arange(len(labels_p)))
    ax.set_xticklabels(labels_p, rotation=45, ha='right')
    ax.set_yticklabels(labels_p)
    plt.title("p-values between Classifiers")
    plt.colorbar(im, ax=ax, label="p-value")
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_path, f"p_values_{title_suffix}.png")) if save_path else None



# --- Step 2: compute statistics and find significant differences ---

# 1. Find the set of pipelines present in each dataset
pipelines_per_dataset = results_all.groupby('dataset')['pipeline'].unique()

# 2. Find the intersection (pipelines present in all datasets)
common_pipelines = set(pipelines_per_dataset.iloc[0])
for arr in pipelines_per_dataset[1:]:
    common_pipelines &= set(arr)

# 3. Filter the dataframe to keep only rows with these pipelines
filtered_results = results_all[results_all['pipeline'].isin(common_pipelines)]

# 4. For each classifier pair, a one-tailed paired-sample permutation test with 
#    the mean difference of per-subject average accuracy as the test statistic
stats = compute_dataset_statistics(filtered_results)

# 5. Extract significant p-values and T-values as matrices
sig_pval, T = find_significant_differences(stats)

# Some prints.
print("Results permutation test:\n", stats, "\n")
print("sig_val:\n", sig_pval, "\n")
print("T:\n", T)

# 6. Plot and save
title_suffix = "_".join(results_all['dataset'].unique()) + "_" + classes_string
plot_save_significant_differences(sig_pval, T, title_suffix=title_suffix, save_path=save_path)


"""
# Results per dataset:
datasets_moabb_name = results_all['dataset'].unique()
for dataset in datasets_moabb_name:
    print(dataset)
    results_dataset = results_all[results_all['dataset'] == dataset]
    stats_dataset = compute_dataset_statistics(results_dataset)
    sig_pval_dataset, T_dataset = find_significant_differences(stats_dataset)
    print(f"\nDataset: {dataset}")
    print("sig_val:\n", sig_pval_dataset)
    print("T:\n", T_dataset)
    title_suffix = dataset + "_" + classes_string
    plot_save_significant_differences(sig_pval_dataset, T_dataset, title_suffix=title_suffix, save_path=save_path)
"""