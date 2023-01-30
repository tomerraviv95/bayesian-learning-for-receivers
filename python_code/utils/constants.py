from enum import Enum

HALF = 0.5


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'


class DetectorType(Enum):
    end_to_end_model = 'end_to_end_model'
    seq_model = 'seq_model'
    model_based_bayesian = 'model_based_bayesian'
    bayesian = 'bayesian'
    black_box = 'black_box'
    bayesian_black_box = 'bayesian_black_box'


class ModulationType(Enum):
    BPSK = 'BPSK'
    QPSK = 'QPSK'
    EightPSK = 'EightPSK'
