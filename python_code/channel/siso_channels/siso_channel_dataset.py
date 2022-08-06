from typing import Tuple

import numpy as np
import torch
from numpy.random import default_rng

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MEMORY_LENGTH, MODULATION_NUM_MAPPING
from python_code.channel.modulator import MODULATION_DICT
from python_code.channel.siso_channels.cost_siso_channel import Cost2100SISOChannel
from python_code.channel.siso_channels.isi_awgn_channel import ISIAWGNChannel
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModels, ModulationType
from python_code.utils.trellis_utils import calculate_siso_states, \
    break_transmitted_siso_word_to_symbols

conf = Config()

SISO_CHANNELS_DICT = {ChannelModels.Synthetic.name: ISIAWGNChannel,
                      ChannelModels.Cost2100.name: Cost2100SISOChannel}


class SISOChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = MEMORY_LENGTH
        self.h_shape = [1, MEMORY_LENGTH]
        self.rx_length = 1

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # create pilots and data
        b_pilots = self._generate_all_classes_pilots()
        b_data = self._bits_generator.integers(0, 2, size=(1, self._block_length - self._pilots_length))
        b = np.concatenate([b_pilots, b_data], axis=1).reshape(1, -1)
        # add zero bits
        padded_b = np.concatenate(
            [np.zeros([b.shape[0], MEMORY_LENGTH - 1]), b, np.zeros([b.shape[0], MEMORY_LENGTH])], axis=1)
        if conf.modulation_type == ModulationType.QPSK.name:
            raise ValueError("Did not implement the QPSK constellation for the SISO case, switch to BPSK or MIMO!")
        # modulation
        s = MODULATION_DICT[conf.modulation_type].modulate(padded_b)
        # transmit through noisy channel
        rx = SISO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        symbols, rx = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, b), rx.T
        return symbols[:-MEMORY_LENGTH + 1], rx[:-MEMORY_LENGTH + 1]

    def _generate_all_classes_pilots(self):
        tx_pilots = self._bits_generator.integers(0, 2, size=(1, self._pilots_length)).reshape(1, -1)
        b_pilots_by_symbols = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, tx_pilots)
        states = calculate_siso_states(MEMORY_LENGTH,
                                       torch.Tensor(b_pilots_by_symbols[:-MEMORY_LENGTH + 1]).to(DEVICE)).cpu().numpy()
        n_unique = MODULATION_NUM_MAPPING[conf.modulation_type] ** MEMORY_LENGTH
        if len(np.unique(states)) < n_unique:
            return self._generate_all_classes_pilots()
        return tx_pilots

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        # transmit through noisy channel
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = ISIAWGNChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100SISOChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
        else:
            raise ValueError("No such channel model!!!")
        tx, rx = self._transmit(h, snr)
        return tx, h, rx
