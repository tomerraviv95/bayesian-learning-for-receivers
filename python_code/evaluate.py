import os

from python_code.detectors.bayesian_deepsic.bayesian_deep_sic_trainer import \
    BayesianDeepSICTrainer
from python_code.detectors.end_to_end_deepsic.end_to_end_deep_sic_trainer import EndToEndDeepSICTrainer
from python_code.detectors.model_based_bayesian_deepsic.model_based_bayesian_deep_sic_trainer import \
    ModelBasedBayesianDeepSICTrainer
from python_code.detectors.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, DetectorType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

conf = Config()

CHANNEL_TYPE_TO_TRAINER_DICT = {ChannelModes.MIMO.name:
                                    {DetectorType.seq_model.name: SeqDeepSICTrainer,
                                     DetectorType.end_to_end_model.name: EndToEndDeepSICTrainer,
                                     DetectorType.model_based_bayesian.name: ModelBasedBayesianDeepSICTrainer,
                                     DetectorType.black_box_based_bayesian.name: BayesianDeepSICTrainer},
                                }

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.channel_type][conf.detector_type]()
    print(trainer)
    trainer.evaluate()
