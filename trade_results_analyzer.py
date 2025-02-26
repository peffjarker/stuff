#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import matplotlib.gridspec as gridspec

from datastr import data_str  # assumes data_str is defined in datastr.py

# Allow evaluation of np.float64 as a built-in float.
np.float64 = float

# Parse each dictionary from the data string.
data_list = []
for line in data_str.strip().splitlines():
    try:
        d = eval(line, {"np": np, "__builtins__": None})
        data_list.append(d)
    except Exception as e:
        print("Error evaluating line:", line)
        print(e)

if not data_list:
    print("No valid data to analyze.")
    exit(1)

# Extract metrics from the data set.
profits = [d['profit'] for d in data_list]
up_trend_rsi_entry = [d['up_trend_rsi_entry'] for d in data_list]
down_trend_rsi_exit = [d['down_trend_rsi_exit'] for d in data_list]
rsi_exit_level = [d['rsi_exit_level'] for d in data_list]

# Print a simple analysis report.
print("===== Analysis Report =====")
print("Number of strategies analyzed:", len(data_list))
print("Average profit: {:.2f}".format(sum(profits) / len(profits)))
for idx, d in enumerate(data_list):
    print(f"Strategy {idx}: up_trend_rsi_entry={d['up_trend_rsi_entry']}, "
          f"down_trend_rsi_exit={d['down_trend_rsi_exit']}, "
          f"rsi_exit_level={d['rsi_exit_level']}")

# -----------------------------
# 3D Scatter Plot with External Info Panel and Zooming
# -----------------------------
# Create a figure with two subplots: one for the 3D scatter, one for details.
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax = fig.add_subplot(gs[0], projection='3d')
ax_text = fig.add_subplot(gs[1])
ax_text.axis('off')  # hide the axis for the text panel

# Create the scatter plot with picker enabled.
sc = ax.scatter(up_trend_rsi_entry, down_trend_rsi_exit, rsi_exit_level,
                c=profits, cmap='viridis', s=100, edgecolors='k', picker=True)

ax.set_xlabel('Up Trend RSI Entry')
ax.set_ylabel('Down Trend RSI Exit')
ax.set_zlabel('RSI Exit Level')
ax.set_title('Profit vs. RSI Tuple')

# Add a color bar to indicate profit.
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Profit')

# Define the pick event handler to update the text panel.
def on_pick(event):
    ind = event.ind  # list of indices of points picked
    i = ind[0]       # select the first point if multiple are picked
    info = (
        f"Strategy {i}:\n"
        f"  Up Trend RSI Entry: {up_trend_rsi_entry[i]}\n"
        f"  Down Trend RSI Exit: {down_trend_rsi_exit[i]}\n"
        f"  RSI Exit Level: {rsi_exit_level[i]}\n"
        f"  Profit: {profits[i]}"
    )
    print(info)  # prints to console as well

    # Update the external text panel.
    ax_text.clear()
    ax_text.text(0.05, 0.5, info, fontsize=12, verticalalignment='center',
                 transform=ax_text.transAxes)
    ax_text.axis('off')
    fig.canvas.draw_idle()

# Define a scroll event handler for zooming.
def on_scroll(event):
    base_scale = 1.1  # zoom factor
    # event.button is 'up' for zoom in and 'down' for zoom out.
    if event.button == 'up':
        ax.dist /= base_scale
    elif event.button == 'down':
        ax.dist *= base_scale
    fig.canvas.draw_idle()

# Connect the event handlers.
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.tight_layout()
plt.show()
