from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.channel.mimo_channels.cost_mimo_channel import Cost2100MIMOChannel
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModels

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
        tx_pilots = self._bits_generator.integers(0, BPSKModulator.constellation_size,
                                                  size=(self._pilots_length, N_USER))
        tx_data = self._bits_generator.integers(0, BPSKModulator.constellation_size,
                                                size=(self._block_length - self._pilots_length, N_USER))
        tx = np.concatenate([tx_pilots, tx_data])
        # modulation
        s = BPSKModulator.modulate(tx.T)
        # pass through channel
        rx = MIMO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr)
        return tx, rx.T

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
