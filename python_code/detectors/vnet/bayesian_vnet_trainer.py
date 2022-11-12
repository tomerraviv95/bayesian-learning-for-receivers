import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.trainer import Trainer
from python_code.detectors.vnet.vnet_detector import VNETDetector
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_siso_states

conf = Config()
EPOCHS = 500


class VNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_states = BPSKModulator.constellation_size ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.lr = 1e-3
        self.probs_vec = None
        super().__init__()

    def __str__(self):
        return 'ViterbiNet'

    def _initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = VNETDetector(n_states=self.n_states)

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
        # now Bayesian loss
        data_fitting_loss_term_ARM_ori = self.criterion(input=est[1], target=gt_states)
        data_fitting_loss_term_ARM_tilde = self.criterion(input=est[2], target=gt_states)
        ARM_delta = (data_fitting_loss_term_ARM_tilde-data_fitting_loss_term_ARM_ori)
        grad_logit1 = torch.sum(ARM_delta*(est[4][0]-0.5))
        grad_logit2 = torch.sum(ARM_delta*(est[4][1]-0.5))
        arm_loss = grad_logit1*self.detector.net.dropout_logit1 + grad_logit2*self.detector.net.dropout_logit2 # this way, we can simply use backward()
        kl_term /= TOTAL_BATCH_SIZE # Tomer: can you change this to proper variable please ? 
        loss = data_fitting_loss_term + kl_term + arm_loss
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase='val')
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
            info_for_Bayesian_training = self.detector(rx.float(), phase='train')
            #soft_estimation = info_for_Bayesian_training[0]
            current_loss = self.run_train_loop(est=info_for_Bayesian_training, tx=tx)
            loss += current_loss
