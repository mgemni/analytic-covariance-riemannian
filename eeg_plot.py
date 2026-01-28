import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================
# ===== Load data =====
# =====================

# Data path
results_path = "./results_eeg"

# Pipelines
pipeline_names_all = {
            "COV+MDM": "COV+MDM",
            "ACOV+MDM": "ACOV+MDM",
            "COV+TSP+LR": "COV+TS+LR",
            "ACOV+TSH+LR": "ACOV+TSH+LR",
            }

# If we only want a subset of pipelines, set this here.
pipeline_names = pipeline_names_all

# Load results for each classifier/pipeline (all subjects) to a dict.
pipeline_results = {}
pipeline_results_avg_per_subject = {}
for clf_name in pipeline_names.keys():

    file_path = os.path.join(results_path, "results_{}.csv".format(clf_name))
    #print(file_path)

    # Read .csv and extract data
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        score_column = df["score"]
        print("Average score for {}: \t".format(clf_name), np.mean(score_column))
        pipeline_results[clf_name] = score_column

        # Average score per user:
        average_scores = df.groupby("subject")["score"].mean().reset_index()
        pipeline_results_avg_per_subject[clf_name] = average_scores["score"].to_numpy()

    else:
        print("{}:".format(clf_name))

data = pipeline_results_avg_per_subject
print(data.values())


# =====================
# ===== Plot data =====
# =====================

# Number of bars per group
S = len(next(iter(data.values())))
groups = list(data.keys())
num_groups = len(groups)

# X positions
bar_width = 0.2
indices = np.arange(S)

# Colors from yellow-green-blue scale (viridis or similar)
cmap = plt.colormaps['viridis']

# Pick colors: lighter (yellow-green) and darker (green-blue)
color1 = cmap(0.1)  # more yellow-green
color2 = cmap(0.5)  # mid green
color3 = cmap(0.7)  # deeper green
color4 = cmap(0.95)  # bluish-green
colors = [color1, color2, color3, color4]

# Patterns
patterns = [
    "",        # no hatch
    "\\\\\\",     # diagonal lines
    "",    # diagonal lines the other way
    "\\\\\\",      # cross
    "xx",     # dotted
]

# Adjust hatch line thickness if needed (e.g., thinner lines)
plt.rcParams["hatch.linewidth"] = 0.5

fig, ax = plt.subplots(figsize=(8, 6))
# Plot each group
for i, group in enumerate(groups):
    bar_positions = indices + i * bar_width
    
    # Get a color/hatch from the list, cycling if needed
    color = colors[i % len(colors)]
    hatch = patterns[i % len(patterns)]
    
    ax.bar(
        bar_positions,
        data[group],
        bar_width,
        label=pipeline_names[group],
        color=color,
        edgecolor="black",
        hatch=hatch
    )

# Labeling
ax.set_xlabel('Subject', fontsize=14)
ax.set_ylabel('Average accuracy', fontsize=14)

ax.set_title('')
ax.set_xticks(indices + bar_width * (num_groups - 1) / 2)
ax.set_xticklabels([f'S{i+1}' for i in range(S)])
ax.tick_params(axis='both', labelsize=12)

#ax.legend(loc='lower right')
ax.legend(fontsize=13, loc='lower right')
ax.set_ylim(bottom=0.4)

plt.tight_layout()
#plt.show()

# Save as EPS or SVG
fig.savefig(results_path + "/EEG_results.eps")
fig.savefig(results_path + "/EEG_results.svg", bbox_inches='tight', transparent=True)

# --- See what the plot looks like in gray-scale ---
"""
# Render the figure to an image array
fig.canvas.draw()
img = np.array(fig.canvas.renderer.buffer_rgba())

# Convert to grayscale
gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0  # Luminance formula, normalize

# Display the grayscale version
plt.figure(figsize=(12, 6))
plt.imshow(gray, cmap='gray', aspect='auto')
plt.axis('off')
plt.show()"
"""