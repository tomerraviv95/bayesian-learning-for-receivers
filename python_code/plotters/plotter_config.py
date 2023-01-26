from enum import Enum
from typing import Tuple, List, Dict

import numpy as np

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    MIMO_BY_SNR_FADING_LINEAR = 'MIMO_BY_SNR_FADING_LINEAR'
    MIMO_BY_RELIABILITY_FADING_LINEAR = 'MIMO_BY_RELIABILITY_FADING_LINEAR'
    MIMO_BY_ECE_LINEAR = 'MIMO_BY_ECE_LINEAR'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, list, str, str]:
    # figure 1
    if plot_type == PlotType.MIMO_BY_SNR_FADING_LINEAR:
        params_dicts = [
            {'snr': 6, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 7, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 8, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'block_length': 20750, 'pilot_size': 750},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 7, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 9, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 11, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 13, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 7, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 11, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 13, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 7, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 11, 'detector_type': DetectorType.model_based_bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 13, 'detector_type': DetectorType.model_based_bayesian.name,
             'channel_type': ChannelModes.MIMO.name},

        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(6, 14))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.MIMO_BY_ECE_LINEAR:
        params_dicts = [
            {'snr': 6, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 7, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 9, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 7, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 7, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 6, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 7, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 8, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(6, 14))
        xlabel, ylabel = 'SNR', 'ECE'
    elif plot_type == PlotType.MIMO_BY_RELIABILITY_FADING_LINEAR:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name,
             'channel_type': ChannelModes.MIMO.name},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name},
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=11)
        xlabel, ylabel = 'Confidence', 'Accuracy/Confidence'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
