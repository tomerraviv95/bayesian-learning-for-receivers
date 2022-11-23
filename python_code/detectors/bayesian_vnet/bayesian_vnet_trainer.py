import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.bayesian_vnet.bayesian_vnet_detector import BayesianVNETDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
from python_code.utils.trellis_utils import calculate_siso_states

conf = Config()
EPOCHS = 500


class BayesianVNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_states = BPSKModulator.constellation_size ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.lr = 5e-3
        self.probs_vec = None
        self.ensemble_num = 5
        self.kl_beta = 1e-1
        super().__init__()

    def __str__(self):
        return 'Bayesian ViterbiNet'

    def _initialize_detector(self):
        """
        Loads the Bayesian ViterbiNet detector
        """
        self.detector = BayesianVNETDetector(n_states=self.n_states,
                                             length_scale=0.1,
                                             ensemble_num=self.ensemble_num)

    def calc_loss(self, est, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est[0]: [1,transmission_length,n_states], each element is a probability
        :est[1]: log_prob_ARM_ori
        :est[2]: log_prob_ARM_tilde
        :est[3]: kl_term
        :est[4]: (u1, u2)
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_siso_states(self.memory_length, tx)
        data_fitting_loss_term = self.criterion(input=est[0], target=gt_states)

        # ARM Loss
        arm_loss = 0
        for i in range(self.ensemble_num):
            data_fitting_loss_term_ARM_ori = self.criterion(input=est[1][i], target=gt_states)
            data_fitting_loss_term_ARM_tilde = self.criterion(input=est[2][i], target=gt_states)
            ARM_delta = (data_fitting_loss_term_ARM_tilde - data_fitting_loss_term_ARM_ori)
            grad_logit = ARM_delta * (est[3][i] - 0.5)
            arm_loss += torch.matmul(grad_logit, self.detector.net.dropout_logit.T)
        arm_loss = torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est[4] / tx.shape[0]
        loss = data_fitting_loss_term + arm_loss + kl_term
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase=Phase.TEST)
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        :param h: channel coefficients
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.deep_learning_setup()

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            info_for_Bayesian_training = self.detector(rx.float(), phase=Phase.TRAIN)
            current_loss = self.run_train_loop(est=info_for_Bayesian_training, tx=tx)
            loss += current_loss
