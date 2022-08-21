from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    Model_VS_Ensemble_By_SNR = 'Model_VS_Ensemble_By_SNR'
    Model_VS_Ensemble_By_Block = 'Model_VS_Ensemble_By_Block'


def get_config(label_name: str) -> Tuple[List[Dict], list, list, str, str, str]:
    # figure 1
    if label_name == PlotType.Model_VS_Ensemble_By_SNR.name:
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
            {'snr': 9, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 11, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 12, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 13, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    # figure 2
    elif label_name == PlotType.Model_VS_Ensemble_By_Block.name:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(1, 101))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_by_blocks'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel, plot_type
