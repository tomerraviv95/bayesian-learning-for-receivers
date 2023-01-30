import os

from python_code import conf
from python_code.detectors.deepsic.bayesian_deepsic.bayesian_deep_sic_trainer import BayesianDeepSICTrainer
from python_code.detectors.deepsic.end_to_end_deepsic.end_to_end_deep_sic_trainer import EndToEndDeepSICTrainer
from python_code.detectors.deepsic.model_based_bayesian_deepsic.model_based_bayesian_deep_sic_trainer import \
    ModelBasedBayesianDeepSICTrainer
from python_code.detectors.deepsic.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer
from python_code.detectors.dnn.bayesian_dnn.bayesian_dnn_trainer import BayesianDNNTrainer
from python_code.detectors.dnn.dnn_trainer import DNNTrainer
from python_code.utils.constants import DetectorType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


CHANNEL_TYPE_TO_TRAINER_DICT = {DetectorType.seq_model.name: SeqDeepSICTrainer,
                                DetectorType.end_to_end_model.name: EndToEndDeepSICTrainer,
                                DetectorType.model_based_bayesian.name: ModelBasedBayesianDeepSICTrainer,
                                DetectorType.bayesian.name: BayesianDeepSICTrainer,
                                DetectorType.black_box.name: DNNTrainer,
                                DetectorType.bayesian_black_box.name: BayesianDNNTrainer}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.detector_type]()
    print(trainer)
    trainer.evaluate()
