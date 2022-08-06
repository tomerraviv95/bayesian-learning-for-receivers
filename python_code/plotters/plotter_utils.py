import datetime
import os
from typing import List, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dir_definitions import FIGURES_DIR, PLOTS_DIR
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.python_utils import load_pkl, save_pkl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

conf = Config()

MIN_BER_COEF = 0.2
MARKER_EVERY = 5


def get_linestyle(method_name: str) -> str:
    if 'ViterbiNet' in method_name or 'DeepSIC' in method_name:
        return 'solid'
    elif 'RNN' in method_name or 'DNN' in method_name:
        return 'dashed'
    else:
        raise ValueError('No such detector!!!')


def get_marker(method_name: str) -> str:
    if 'Regular Training' in method_name:
        return '.'
    elif 'FK Genie' in method_name:
        return 'X'
    elif 'Geometric' in method_name:
        return '>'
    elif 'Translation' in method_name:
        return '<'
    elif 'Rotation' in method_name:
        return 'v'
    elif 'Combined' in method_name:
        return 'D'
    elif 'Extended Pilot Training' in method_name:
        return 'o'
    else:
        raise ValueError('No such method!!!')


def get_color(method_name: str) -> str:
    if 'Regular Training' in method_name:
        return 'b'
    elif 'FK Genie' in method_name:
        return 'black'
    elif 'Geometric' in method_name:
        return 'orange'
    elif 'Translation' in method_name:
        return 'pink'
    elif 'Rotation' in method_name:
        return 'green'
    elif 'Combined' in method_name:
        return 'red'
    elif 'Extended Pilot Training' in method_name:
        return 'royalblue'
    else:
        raise ValueError('No such method!!!')


def get_ser_plot(dec: Trainer, run_over: bool, method_name: str, trial=None):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name, str(conf.channel_type)])
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        ser_total = load_pkl(plots_path)
    else:
        # otherwise - run again
        print("calculating fresh")
        ser_total = dec.evaluate()
        save_pkl(plots_path, ser_total)
    print(ser_total)
    return ser_total


def plot_by_values(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], values: List[float], xlabel: str,
                   ylabel: str):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    names = []
    for i in range(len(all_curves)):
        if all_curves[i][1] not in names:
            names.append(all_curves[i][1])

    cur_name, mean_sers_dict = populate_mean_sers_dict(all_curves, names)

    # plots all methods
    for method_name in names:
        plt.plot(values, mean_sers_dict[method_name], label=method_name,
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2)

    plt.xticks(values, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    trainer_name = cur_name.split(' ')[0]
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_snrs_{trainer_name}.png'),
                bbox_inches='tight')
    plt.show()


def populate_mean_sers_dict(all_curves: List[Tuple[float, str]], names: List[str]) -> Tuple[
    str, Dict[str, List[np.ndarray]]]:
    mean_sers_dict = {}
    for method_name in names:
        mean_sers = []
        for ser, cur_name in all_curves:
            mean_ser = np.mean(ser)
            if cur_name != method_name:
                continue
            mean_sers.append(mean_ser)
        mean_sers_dict[method_name] = mean_sers
    return cur_name, mean_sers_dict
