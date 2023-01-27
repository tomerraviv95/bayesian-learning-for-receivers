from enum import Enum

HALF = 0.5
QUARTER = 0.25
C = 0.5
H_COEF = 0.8

class Phase(Enum):
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'


class ChannelModes(Enum):
    SISO = 'SISO'
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'
    Cost2100 = 'Cost2100'


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
