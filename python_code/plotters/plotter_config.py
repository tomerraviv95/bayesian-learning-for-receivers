from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    Model_VS_Ensemble_By_Block_Synth_Linear = 'Model_VS_Ensemble_By_Block_Synth_Linear'
    Model_VS_Ensemble_By_SNR_Synth_Linear = 'Model_VS_Ensemble_By_SNR_Synth_Linear'
    Model_VS_Ensemble_By_Block_Synth_Non_Linear = 'Model_VS_Ensemble_By_Block_Synth_Non_Linear'
    Model_VS_Ensemble_By_SNR_Synth_Non_Linear = 'Model_VS_Ensemble_By_SNR_Synth_Non_Linear'
    Model_VS_Ensemble_By_Block_Cost2100 = 'Model_VS_Ensemble_By_Block_Cost2100'
    Model_VS_Ensemble_By_SNR_Cost2100 = 'Model_VS_Ensemble_By_SNR_Cost2100'


def get_config(label_name: str) -> Tuple[List[Dict], list, list, str, str, str]:
    # figure 1a
    if label_name == PlotType.Model_VS_Ensemble_By_Block_Synth_Linear.name:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(1, 101))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_by_blocks'
    # figure 1b
    elif label_name == PlotType.Model_VS_Ensemble_By_SNR_Synth_Linear.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 9, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 11, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 12, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
            {'snr': 13, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    # figure 2a
    elif label_name == PlotType.Model_VS_Ensemble_By_Block_Synth_Non_Linear.name:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(1, 101))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_by_blocks'
    # figure 1b
    elif label_name == PlotType.Model_VS_Ensemble_By_SNR_Synth_Non_Linear.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 9, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 11, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 12, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
            {'snr': 13, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'linear': False},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    # figure 3a
    elif label_name == PlotType.Model_VS_Ensemble_By_Block_Cost2100.name:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(1, 101))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_by_blocks'
    # figure 3b
    elif label_name == PlotType.Model_VS_Ensemble_By_SNR_Cost2100.name:
        params_dicts = [
            {'snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 9, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 10, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 11, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 12, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
            {'snr': 13, 'detector_type': DetectorType.ensemble.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'channel_model': 'Cost2100'},
        ]
        methods_list = [
            'Regular'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
        plot_type = 'plot_by_snrs'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel, plot_type
