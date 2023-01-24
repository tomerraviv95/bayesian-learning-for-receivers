import os
from collections import namedtuple
from typing import Tuple, List, Dict, Union

from dir_definitions import CONFIG_RUNS_DIR
from python_code.detectors.trainer import Trainer
from python_code.evaluate import CHANNEL_TYPE_TO_TRAINER_DICT
from python_code.plotters.plotter_utils import get_all_plots
from python_code.utils.config_singleton import Config

RunParams = namedtuple(
    "RunParams",
    "run_over trial_num",
    defaults=[False, 1]
)

BER_ERROR_THRESHOLD = 0.4


def set_method_name(conf: Config, method_name: str, params_dict: Dict[str, Union[int, str]]) -> str:
    """
    Set values of params dict to current config. And return the field and their respective values as the name of the run,
    used to save as pkl file for easy access later.
    :param conf: config file.
    :param method_name: the desired augmentation scheme name
    :param params_dict: the run params
    :return: name of the run
    """
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)
    return name


def gather_plots_by_trials(all_curves: List[Tuple[str, List, List, List]], conf: Config, method_name: str,
                           name: str, run_over: bool,
                           trial_num: int, trainer: Trainer):
    """
    Run the experiments #trial_num times, averaging over the whole run's aggregated ser.
    """
    total_ber = []
    total_correct_values_list = []
    total_error_values_list = []
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        trainer.__init__()
        ber, correct_values_list, error_values_list = get_all_plots(trainer, run_over=run_over,
                                                                    method_name=method_name + name,
                                                                    trial=trial)
        errored_indices = [ind for ind in range(len(ber)) if ber[ind] > BER_ERROR_THRESHOLD]
        if len(errored_indices) > 0:
            for errored_index in errored_indices[::-1]:
                del ber[errored_index]
                del correct_values_list[errored_index]
                del error_values_list[errored_index]
        total_ber.append(ber)
        total_correct_values_list.extend(correct_values_list)
        total_error_values_list.extend(error_values_list)
    all_curves.append((method_name, total_ber, total_correct_values_list, total_error_values_list))


def compute_for_method(all_curves: List[Tuple[float, str]], method: str, params_dict: Dict[str, Union[int, str]],
                       run_params_obj: RunParams):
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, params_dict['channel_type'], f'{method}.yaml'))
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[params_dict['channel_type']][params_dict['detector_type']]()
    full_method_name = f'{trainer.__str__()}'
    print(full_method_name)
    name = set_method_name(conf, full_method_name, params_dict)
    gather_plots_by_trials(all_curves, conf, full_method_name, name, run_params_obj.run_over, run_params_obj.trial_num,
                           trainer)
