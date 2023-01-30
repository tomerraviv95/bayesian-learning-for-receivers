from typing import Tuple

import numpy as np
import torch
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import MODULATION_DICT, MODULATION_NUM_MAPPING
from python_code.utils.constants import ModulationType
from python_code.utils.trellis_utils import get_bits_from_qpsk_symbols, get_bits_from_eightpsk_symbols
from python_code.utils.trellis_utils import get_qpsk_symbols_from_bits, generate_bits_by_state, \
    get_eightpsk_symbols_from_bits


class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = conf.n_user
        self.h_shape = [conf.n_ant, conf.n_user]
        self.rx_length = conf.n_ant

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        tx_pilots = self._generate_all_classes_pilots()
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, conf.n_user))
        tx = np.concatenate([tx_pilots, tx_data])
        # modulation
        s = MODULATION_DICT[conf.modulation_type].modulate(tx.T)
        # pass through channel
        rx = SEDChannel.transmit(s=s, h=h, snr=snr)
        if conf.modulation_type == ModulationType.QPSK.name:
            tx = get_qpsk_symbols_from_bits(tx)
        if conf.modulation_type == ModulationType.EightPSK.name:
            tx = get_eightpsk_symbols_from_bits(tx)
        return tx, rx.T

    def _generate_all_classes_pilots(self):
        # generate random pilots block of bits
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, conf.n_user))
        if conf.modulation_type == ModulationType.QPSK.name:
            tx_pilots = get_qpsk_symbols_from_bits(tx_pilots)
        if conf.modulation_type == ModulationType.EightPSK.name:
            tx_pilots = get_eightpsk_symbols_from_bits(tx_pilots)

        # ensure that you have each state
        unique_states_num = MODULATION_NUM_MAPPING[conf.modulation_type] ** conf.n_user
        for unique_state in range(min(unique_states_num, tx_pilots.shape[0])):
            tx_pilots[unique_state] = generate_bits_by_state(unique_state, conf.n_user).cpu().numpy().reshape(-1)

        if conf.modulation_type == ModulationType.QPSK.name:
            tx_pilots = get_bits_from_qpsk_symbols(torch.Tensor(tx_pilots)).cpu().numpy()
        if conf.modulation_type == ModulationType.EightPSK.name:
            tx_pilots = get_bits_from_eightpsk_symbols(torch.Tensor(tx_pilots)).cpu().numpy()
        return tx_pilots

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
        tx, rx = self._transmit(h, snr)
        return tx, h, rx
