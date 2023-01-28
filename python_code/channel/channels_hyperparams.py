import math

from python_code.utils.config_singleton import Config

conf = Config()

# MIMO
N_USER = 4
N_ANT = 4

MODULATION_NUM_MAPPING = {
    'BPSK': 2,
    'QPSK': 4,
    'EightPSK': 8
}

CONSTELLATION_BITS = int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type]))
