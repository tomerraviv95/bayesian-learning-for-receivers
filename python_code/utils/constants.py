from enum import Enum

HALF = 0.5
QUARTER = 0.25
C = 0.5
H_COEF = 0.8
TRAIN_VAL_SPLIT_RATIO = 1


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
    black_box = 'black_box'
    model = 'model'
    bayesian = 'bayesian'
