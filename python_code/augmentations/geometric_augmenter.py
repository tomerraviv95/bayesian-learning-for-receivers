from typing import Tuple

import torch

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

conf = Config()


class GeometricAugmenter:
    """
    A proposed augmentations scheme. Based on the calculated centers and variances for each class, it draws samples.
    """

    def __init__(self, centers: torch.Tensor, stds: torch.Tensor, n_states: int, state_size: int,
                 gt_states: torch.Tensor):
        super().__init__()
        self._centers = centers
        self._stds = stds
        self._n_states = n_states
        self._state_size = state_size
        self._gt_states = gt_states

    def augment(self, rx: torch.Tensor, tx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.channel_type == ChannelModes.SISO.name:
            to_augment_state = calculate_siso_states(MEMORY_LENGTH, tx)[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            to_augment_state = calculate_mimo_states(N_USER, tx)[0]
        else:
            raise ValueError("No such channel type!!!")

        if conf.channel_type == ChannelModes.SISO.name:
            rx = self._centers[to_augment_state] + self._stds[to_augment_state] * torch.randn(
                [1, self._state_size]).to(DEVICE)
        elif conf.channel_type == ChannelModes.MIMO.name:
            rx = self._centers[to_augment_state] + self._stds[to_augment_state] * torch.randn(
                self._centers[to_augment_state].shape).to(DEVICE)
            rx = rx.unsqueeze(0)
        else:
            raise ValueError("No such channel type!!!")
        return rx, tx

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    @property
    def stds(self) -> torch.Tensor:
        return self._stds
