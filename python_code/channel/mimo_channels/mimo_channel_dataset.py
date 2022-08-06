from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code.channel.channels_hyperparams import N_ANT, N_USER, MODULATION_NUM_MAPPING
from python_code.channel.mimo_channels.cost_mimo_channel import Cost2100MIMOChannel
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import MODULATION_DICT
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModels, ModulationType
from python_code.utils.trellis_utils import get_qpsk_symbols_from_bits, generate_bits_by_state

conf = Config()

MIMO_CHANNELS_DICT = {ChannelModels.Synthetic.name: SEDChannel,
                      ChannelModels.Cost2100.name: Cost2100MIMOChannel}


class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = N_USER
        self.h_shape = [N_ANT, N_USER]
        self.rx_length = N_ANT

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        tx_pilots = self._generate_all_classes_pilots()
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, N_USER))
        tx = np.concatenate([tx_pilots, tx_data])
        # modulation
        s = MODULATION_DICT[conf.modulation_type].modulate(tx.T)
        # pass through channel
        rx = MIMO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr)
        if conf.modulation_type == ModulationType.QPSK.name:
            tx = get_qpsk_symbols_from_bits(tx)
        return tx, rx.T

    def _generate_all_classes_pilots(self):
        # generate random pilots block of bits
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, N_USER))
        if conf.modulation_type == ModulationType.QPSK.name:
            tx_pilots = get_qpsk_symbols_from_bits(tx_pilots)

        # ensure that you have each state
        unique_states_num = MODULATION_NUM_MAPPING[conf.modulation_type] ** N_USER
        for unique_state in range(min(unique_states_num, tx_pilots.shape[0])):
            tx_pilots[unique_state] = generate_bits_by_state(unique_state, N_USER).cpu().numpy().reshape(-1)

        if conf.modulation_type == ModulationType.QPSK.name:
            first_bit = tx_pilots % 2
            second_bit = np.floor(tx_pilots / 2)
            concat_array = np.concatenate([np.expand_dims(first_bit, -1), np.expand_dims(second_bit, -1)], axis=2)
            transposed_array = np.transpose(concat_array, [0, 2, 1])
            tx_pilots = transposed_array.reshape(2 * first_bit.shape[0], -1)
        return tx_pilots

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = SEDChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100MIMOChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        else:
            raise ValueError("No such channel model!!!")
        tx, rx = self._transmit(h, snr)
        return tx, h, rx
