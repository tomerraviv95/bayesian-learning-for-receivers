from enum import Enum

HALF = 0.5
QUARTER = 0.25


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    SISO = 'SISO'
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'
    Cost2100 = 'Cost2100'


class DetectorType(Enum):
    black_box = 'black_box'
    model = 'model'


class ModulationType(Enum):
    BPSK = 'BPSK'
    QPSK = 'QPSK'
