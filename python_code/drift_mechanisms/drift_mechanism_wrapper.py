import random


class DriftMechanismWrapper:

    def __init__(self, type: str):
        self.drift_mechanism = DRIFT_MECHANISMS_DICT[type]

    def is_train(self, *args):
        return self.drift_mechanism.is_train(*args)


class AlwaysDriftMechanism:
    @staticmethod
    def is_train(*args):
        return True


class RandomDriftMechanism:
    @staticmethod
    def is_train(*args):
        return bool(random.getrandbits(1))


DRIFT_MECHANISMS_DICT = {
    'always': AlwaysDriftMechanism,
    'random': RandomDriftMechanism
}
