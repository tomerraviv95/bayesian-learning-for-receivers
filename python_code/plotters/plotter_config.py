from enum import Enum
from typing import Tuple, List, Dict

import numpy as np

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    BY_BLOCK = 'By_Block'
    BY_SNR = 'By_SNR'
    BY_RELIABILITY = 'By_Reliability'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, list, str, str]:
    # figure 1
    if plot_type == PlotType.BY_SNR:
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
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.BY_BLOCK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(1, 101))
        xlabel, ylabel = 'block_index', 'BER'
    elif plot_type == PlotType.BY_RELIABILITY:
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 50},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 50},
            {'snr': 12, 'detector_type': DetectorType.maximum_likelihood.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 50}
        ]
        methods_list = [
            'Regular'
        ]
        values = np.linspace(start=0, stop=1, num=6)
        xlabel, ylabel = 'Reliability', 'Metric'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
