from python_code.detectors.deepsic.deep_sic_trainer import DeepSICTrainer
from python_code.detectors.dnn.dnn_trainer import DNNTrainer
from python_code.detectors.rnn.rnn_trainer import RNNTrainer
from python_code.detectors.vnet.vnet_trainer import VNETTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, DetectorType

conf = Config()

CHANNEL_TYPE_TO_TRAINER_DICT = {ChannelModes.SISO.name:
                                    {DetectorType.model.name: VNETTrainer,
                                     DetectorType.black_box.name: RNNTrainer},
                                ChannelModes.MIMO.name:
                                    {DetectorType.model.name: DeepSICTrainer,
                                     DetectorType.black_box.name: DNNTrainer},
                                }

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.channel_type][conf.detector_type]()
    print(trainer)
    trainer.evaluate()
