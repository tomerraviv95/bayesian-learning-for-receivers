import datetime
import os
from typing import List, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dir_definitions import FIGURES_DIR, PLOTS_DIR
from python_code.detectors.trainer import Trainer
from python_code.plotters.plotter_config import PlotType
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
    if 'Bayesian' in method_name:
        return '.'
    elif 'ViterbiNet' in method_name:
        return 'X'
    else:
        raise ValueError('No such method!!!')


def get_color(method_name: str) -> str:
    if 'Bayesian' in method_name:
        return 'blue'
    elif 'ViterbiNet' in method_name:
        return 'black'
    else:
        raise ValueError('No such method!!!')


def get_all_plots(dec: Trainer, run_over: bool, method_name: str, trial=None):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name, str(conf.channel_type)])
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name)
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path + '_ber' + '.pkl') and not run_over:
        print("Loading plots")
        ber_total = load_pkl(plots_path, type='ber')
        correct_values_list = load_pkl(plots_path, type='cor')
        error_values_list = load_pkl(plots_path, type='err')
    else:
        # otherwise - run again
        print("Calculating fresh")
        ber_total, correct_values_list, error_values_list = dec.evaluate()
        save_pkl(plots_path, ber_total, type='ber')
        save_pkl(plots_path, correct_values_list, type='cor')
        save_pkl(plots_path, error_values_list, type='err')
    print(ber_total, correct_values_list, error_values_list)
    return ber_total, correct_values_list, error_values_list


def plot_by_values(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], values: List[float], xlabel: str,
                   ylabel: str, plot_type: str):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    names = []
    for i in range(len(all_curves)):
        if all_curves[i][0] not in names:
            names.append(all_curves[i][0])

    cur_name, sers_dict = populate_sers_dict(all_curves, names, plot_type)
    if plot_type == PlotType.BY_BLOCK:
        MARKER_EVERY = 10
        x_ticks = [1].extend(values[MARKER_EVERY - 1::MARKER_EVERY])
        x_labels = [1].extend(values[MARKER_EVERY - 1::MARKER_EVERY])
    elif plot_type == PlotType.BY_SNR:
        MARKER_EVERY = 1
        x_ticks = values
        x_labels = values
    else:
        raise ValueError("No such plot type!")

    # plots all methods
    for method_name in names:
        plt.plot(values, sers_dict[method_name], label=method_name,
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2,
                 markevery=MARKER_EVERY)

    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    trainer_name = cur_name.split(' ')[0]
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_snrs_{trainer_name}.png'),
                bbox_inches='tight')
    plt.show()


def plot_by_values2(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], values: List[float], xlabel: str,
                    ylabel: str, plot_type: str):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    names = []
    for i in range(len(all_curves)):
        if all_curves[i][0] not in names:
            names.append(all_curves[i][0])

    cur_name, sers_dict = populate_sers_dict(all_curves, names, plot_type)
    if plot_type == 'plot_by_blocks':
        MARKER_EVERY = 10
        x_ticks = [1].extend(values[MARKER_EVERY - 1::MARKER_EVERY])
        x_labels = [1].extend(values[MARKER_EVERY - 1::MARKER_EVERY])
    elif plot_type == 'plot_by_snrs':
        MARKER_EVERY = 1
        x_ticks = values
        x_labels = values
    else:
        raise ValueError("No such plot type!")

    # plots all methods
    for method_name in names:
        plt.plot(values, sers_dict[method_name], label=method_name,
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2,
                 markevery=MARKER_EVERY)

    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    trainer_name = cur_name.split(' ')[0]
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_snrs_{trainer_name}.png'),
                bbox_inches='tight')
    plt.show()


def populate_sers_dict(all_curves: List[Tuple[float, str]], names: List[str], plot_type: PlotType) -> Tuple[
    str, Dict[str, List[np.ndarray]]]:
    sers_dict = {}
    for method_name in names:
        sers_list = []
        for cur_name, ser, _, _ in all_curves:
            if cur_name != method_name:
                continue
            if plot_type == PlotType.BY_BLOCK:
                agg_ser = (np.cumsum(ser[0]) / np.arange(1, len(ser[0]) + 1))
                sers_list.extend(agg_ser)
            elif plot_type == PlotType.BY_SNR:
                mean_ser = np.mean(ser)
                sers_list.append(mean_ser)
            else:
                raise ValueError("No such plot type!")
        sers_dict[method_name] = sers_list
    return cur_name, sers_dict
