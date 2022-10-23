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
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
        ]
        methods_list = [
            'ViterbiNet'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif plot_type == PlotType.BY_BLOCK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
        ]
        methods_list = [
            'ViterbiNet'
        ]
        values = list(range(1, 101))
        xlabel, ylabel = 'block_index', 'BER'
    elif plot_type == PlotType.BY_RELIABILITY.name:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
        ]
        methods_list = [
            'ViterbiNet'
        ]
        values = np.linspace(start=0.1, stop=1, step=0.1)
        xlabel, ylabel = 'Reliability', 'Reliability'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
