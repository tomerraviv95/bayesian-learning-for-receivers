import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import scatter

from python_code.channel.channels_hyperparams import N_USER, N_ANT
from python_code.channel.modulator import BPSKModulator

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
colors_dict = {0: 'red', 1: 'orange', 2: 'blue', 3: 'green'}


def online_plotting(tx: torch.Tensor, rx: torch.Tensor, h: torch.Tensor):
    if N_USER != 2 or N_ANT != 2:
        raise ValueError("Only valid for N_user=2 and N_ant=2! Please change in config")
    received_array = rx.cpu().numpy()
    transmitted_array = tx.cpu().numpy()
    color_codings = np.sum(np.array([2, 1]) * transmitted_array, axis=1)
    s = BPSKModulator.modulate(transmitted_array)
    true_received_centers = np.matmul(s, h.cpu().numpy())
    for color_coding in np.unique(color_codings):
        print(colors_dict[color_coding])
        mask = (color_codings == color_coding)
        scatter(x=received_array[mask, 0], y=received_array[mask, 1], marker='o', c=colors_dict[color_coding])
        scatter(x=true_received_centers[mask, 0], y=true_received_centers[mask, 1], marker='x',
                c='black')
    RIGHT_LIM, TOP_LIM = 2, 2
    LEFT_LIM, BOTTOM_LIM = -2, -2
    plt.plot([LEFT_LIM, RIGHT_LIM], [0, 0], c='black')
    plt.plot([0, 0], [BOTTOM_LIM, TOP_LIM], c='black')
    plt.xlim([LEFT_LIM, RIGHT_LIM])
    plt.ylim([BOTTOM_LIM, TOP_LIM])
    plt.xticks([-1, 1])
    plt.yticks([-1, 1])
    plt.show()
