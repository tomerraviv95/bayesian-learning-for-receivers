from enum import Enum
from typing import Tuple, List, Dict

import numpy as np

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    SISO_BY_SNR_STATIC_LINEAR = 'SISO_BY_SNR_STATIC_LINEAR'
    SISO_BY_SNR_STATIC_NON_LINEAR = 'SISO_BY_SNR_STATIC_NON_LINEAR'
    SISO_BY_RELIABILITY_STATIC_LINEAR = 'SISO_BY_RELIABILITY_STATIC_LINEAR'
    SISO_BY_RELIABILITY_STATIC_NON_LINEAR = 'SISO_BY_RELIABILITY_STATIC_NON_LINEAR'
    MIMO_BY_SNR_FADING_LINEAR = 'MIMO_BY_SNR_FADING_LINEAR'
    MIMO_BY_SNR_FADING_NON_LINEAR = 'MIMO_BY_SNR_FADING_NON_LINEAR'
    MIMO_BY_RELIABILITY_FADING_LINEAR = 'MIMO_BY_RELIABILITY_FADING_LINEAR'
    MIMO_BY_RELIABILITY_FADING_NON_LINEAR = 'MIMO_BY_RELIABILITY_FADING_NON_LINEAR'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, list, str, str]:
    # figure 1
    if plot_type == PlotType.SISO_BY_SNR_STATIC_LINEAR:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 11, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 13, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 11, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 13, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 9, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 11, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 12, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 13, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name}
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.SISO_BY_SNR_STATIC_NON_LINEAR:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 11, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 13, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 11, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 13, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 9, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 11, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 12, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 13, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False}
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.SISO_BY_RELIABILITY_STATIC_LINEAR:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=9)
        xlabel, ylabel = 'Reliability', 'Metric'
    elif plot_type == PlotType.SISO_BY_RELIABILITY_STATIC_NON_LINEAR:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},

        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=9)
        xlabel, ylabel = 'Reliability', 'Metric'
    elif plot_type == PlotType.MIMO_BY_SNR_FADING_LINEAR:
        params_dicts = [
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
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.MIMO_BY_SNR_FADING_NON_LINEAR:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.MIMO_BY_RELIABILITY_FADING_LINEAR:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=9)
        xlabel, ylabel = 'Reliability', 'Metric'
    elif plot_type == PlotType.MIMO_BY_RELIABILITY_FADING_NON_LINEAR:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'channel_type': ChannelModes.MIMO.name,
             'linear': False, 'fading_in_channel': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=9)
        xlabel, ylabel = 'Reliability', 'Metric'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
