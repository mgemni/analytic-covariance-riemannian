import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import zoom


# --- Pipelines to plot results for ---
pipeline_name_regular = "COV+MDM"
pipeline_name_ACOV = "ACOV+MDM"

results = {}
results[pipeline_name_regular] = None #Pipeline([('cov', Covariances()), ('mdm', MDM())])
results[pipeline_name_ACOV] = None # Pipeline([('haug', HilbertAugmentation()), ('cov', Covariances()), ('mdm', MDM())])


# --- Load results ---
save_folder = "results_ex2"
data_folder = "dpc_500_nofevals_10_sigma1_0.45_sigma2_0.90_noise_std_0.10_grid_30"
filename = "data"

mu1s_dict = {}
mu2s_dict = {}

for key in results.keys():
    print(key)
    df = pd.read_csv("{}/{}/{}_{}.csv".format(save_folder, data_folder,filename,key), index_col=0)
    mu2s_dict[key] = df.index.to_numpy() # Row names as a list
    mu1s_dict[key] = df.columns.to_numpy().astype(float) # Column names as a list
    results[key] = df.to_numpy()


# ========================
# ===== Plot results =====
# ========================

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# Font sizes for labels and colorbar
label_fs = 14           # axis label fontsize
tick_fs = 12            # tick label fontsize (if overriding) 
cbar_tick_fs = 12       # colorbar tick label fontsize
cbar_label_fs = 12      # colorbar axis label fontsize (if you set a label)
contour_fs = 12         # contour label fontsize

fig_size_x = 8
fig_size_y = 6


# Layout/padding control
tight_pad = 0.12        # padding for fig.tight_layout(pad=...)
save_pad_inches = 0.02  # pad_inches for savefig(..., pad_inches=...)
margin_left = 0.12
margin_right = 0.95
margin_top = 0.95
margin_bottom = 0.12

# Find min value across pipelines
mins = []
for key in results.keys():
    mins.append(np.nanmin(results[key]))
min = np.min(np.array(mins))
min = 0.45 # new

# Parameters for interpolation and ticks
zoom_factor = 2
tick_step = 4

# Number of ticks you want in each direction
num_ticks = 7  

for i, (key, data) in enumerate(results.items()):
    mu1s = mu1s_dict[key]
    mu2s = mu2s_dict[key]
    
    # Interpolate (no extrapolation beyond boundary)
    smooth_data = zoom(data, zoom_factor, order=3, mode='nearest')

    fig, ax = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y))
    extent = [mu1s[0], mu1s[-1], mu2s[0], mu2s[-1]]

    im = ax.imshow(smooth_data, origin='lower', extent=extent, aspect='equal')
    im.set_clim(min, 1)
    ax.grid(False)
    #ax.set_title(key)

    # Contours
    contour_levels = np.linspace(min + 0.05, 1 - 0.05, 4, endpoint=True)
    contours = ax.contour(smooth_data, levels=contour_levels, colors='k', linewidths=0.5, extent=extent)
    ax.clabel(contours, inline=True, fmt="%.2f", fontsize=contour_fs)

    # Labels
    ax.set_xlabel(r'$\mu_2$', fontsize=label_fs)
    ax.set_ylabel(r'$\mu_1$', labelpad=0, fontsize=label_fs)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    # increase tick label size for this axes if desired
    ax.tick_params(axis='both', labelsize=tick_fs)

    # --- Ticks ---
    # Define exact tick positions and labels manually
    xtick_positions = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    #xtick_labels = [r'$-\pi$', r'$-\frac{2\pi}{3}$', r'$-\frac{\pi}{3}$', r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$']
    xtick_labels = [r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$']

    ytick_positions = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    #ytick_labels = [r'$-\pi$', r'$-\frac{2\pi}{3}$', r'$-\frac{\pi}{3}$', r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$']
    ytick_labels = [r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$']

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)


    # Colorbar
    # bring colorbar closer to the main axes: smaller pad -> closer; fraction controls thickness
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.025, fraction=0.046)
    # use a space sign flag so positives get a leading space (keeps widths aligned, no '+')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .2f'))
    # enlarge colorbar tick labels and (if set) the colorbar axis label
    cbar.ax.tick_params(labelsize=cbar_tick_fs)
    try:
        cbar.ax.yaxis.label.set_size(cbar_label_fs)
    except Exception:
        pass

    # Save figure
    fig.tight_layout(pad=tight_pad)
    fig.subplots_adjust(left=margin_left, right=margin_right, top=margin_top, bottom=margin_bottom)
    base = f"{save_folder}/fig_result_smooth_{key}_{data_folder}"
    #fig.savefig(base + ".png", bbox_inches='tight', pad_inches=save_pad_inches, transparent=True)
    fig.savefig(base + ".svg", bbox_inches='tight', pad_inches=save_pad_inches, transparent=True)
    fig.savefig(base + ".eps", bbox_inches='tight', pad_inches=save_pad_inches, transparent=False)



# ===========================
# ===== Plot difference =====
# ===========================

# Get improvement of ACOV-pipelines and smoothen
data = results[pipeline_name_ACOV]-results[pipeline_name_regular]
smooth_data = zoom(data, zoom_factor)

# Figure
fig, ax = plt.subplots(1,1, figsize=(fig_size_x, fig_size_y))
extent = [mu1s[0], mu1s[-1], mu2s[0], mu2s[-1]]
im = ax.imshow(smooth_data, origin='lower', extent=extent, aspect='equal')
ax.grid(False)

# Color-limits
min=-0.05
max = 0.3
im.set_clim(min, max)
#ax.set_title("HPD improvement")

# Add iso-contours
contour_levels = np.linspace(-0.00, max-0.05, 4, endpoint=True)  # Adjust levels as needed
contours = ax.contour(smooth_data, levels=contour_levels, colors='k', linewidths=0.5, extent=extent)

# Add labels to the contour lines
ax.clabel(contours, inline=True, fmt="%.2f", fontsize=contour_fs)  # `fmt` controls the format of the labels

# Set LaTeX-style labels for \sigma_2 and \mu_2 (bigger labels)
ax.set_ylabel(r'$\mu_1$', labelpad=0, fontsize=label_fs)
ax.yaxis.set_label_coords(-0.09, 0.5)
ax.set_xlabel(r'$\mu_2$', fontsize=label_fs)
ax.tick_params(axis='both', labelsize=tick_fs)

# --- Ticks ---
# Define exact tick positions and labels manually
xtick_positions = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
#xtick_labels = [r'$-\pi$', r'$-\frac{2\pi}{3}$', r'$-\frac{\pi}{3}$', r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$']
xtick_labels = [r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$']

ytick_positions = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
#ytick_labels = [r'$-\pi$', r'$-\frac{2\pi}{3}$', r'$-\frac{\pi}{3}$', r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$']
ytick_labels = [r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$']


# Set tick positions and LaTeX labels
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels)

ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels)
ax.tick_params(axis='both', labelsize=tick_fs)


# Colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.025, fraction=0.046)
# use a space sign flag so positives get a leading space (keeps widths aligned, no '+')
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .2f'))
cbar.ax.tick_params(labelsize=cbar_tick_fs)
try:
    cbar.ax.yaxis.label.set_size(cbar_label_fs)
except Exception:
    pass

# Save figure
fig.tight_layout(pad=tight_pad)
fig.subplots_adjust(left=margin_left, right=margin_right, top=margin_top, bottom=margin_bottom)
base = f"{save_folder}/fig_acov_improvement_smooth_{key}_{data_folder}"
#fig.savefig(base + ".png", bbox_inches='tight', pad_inches=save_pad_inches, transparent=True)
fig.savefig(base + ".svg", bbox_inches='tight', pad_inches=save_pad_inches, transparent=True)
fig.savefig(base + ".eps", bbox_inches='tight', pad_inches=save_pad_inches, transparent=False)

