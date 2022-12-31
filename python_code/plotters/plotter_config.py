from enum import Enum
from typing import Tuple, List, Dict

import numpy as np

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    BY_BLOCK = 'BY_BLOCK'
    BY_SNR_STATIC_LINEAR = 'BY_SNR_STATIC_LINEAR'
    BY_SNR_FADING_LINEAR = 'BY_SNR_FADING_LINEAR'
    BY_SNR_STATIC_NON_LINEAR = 'BY_SNR_STATIC_NON_LINEAR'
    BY_RELIABILITY = 'BY_RELIABILITY'
    BY_RELIABILITY_NON_LINEAR = 'BY_RELIABILITY_NON_LINEAR'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, list, str, str]:
    # figure 1
    if plot_type == PlotType.BY_SNR_STATIC_LINEAR:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 11, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 13, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 9, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 11, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 12, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 13, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.BY_SNR_STATIC_NON_LINEAR:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 11, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 13, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
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
             'linear': False},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.BY_SNR_FADING_LINEAR:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 11, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 13, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 9, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 11, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 12, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
            {'snr': 13, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'fading': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.BY_BLOCK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(1, 51))
        xlabel, ylabel = 'block_index', 'BER'
    elif plot_type == PlotType.BY_RELIABILITY:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name}
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=9)
        xlabel, ylabel = 'Reliability', 'Metric'
    elif plot_type == PlotType.BY_RELIABILITY_NON_LINEAR:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False},
            {'snr': 10, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'linear': False}
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=9)
        xlabel, ylabel = 'Reliability', 'Metric'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
