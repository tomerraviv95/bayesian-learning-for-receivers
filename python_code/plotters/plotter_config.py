from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    Always_VS_Random = 'Always_VS_Random'


def get_config(label_name: str) -> Tuple[List[Dict], list, list, str, str]:
    # figure 1
    if label_name == PlotType.Always_VS_Random.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 50},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 50},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 50},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 50},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 50},
        ]
        methods_list = [
            'Always',
            'Random'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
