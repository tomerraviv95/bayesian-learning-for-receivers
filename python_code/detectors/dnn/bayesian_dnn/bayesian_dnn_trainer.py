import torch

from python_code import DEVICE, conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.detectors.dnn.bayesian_dnn.bayesian_dnn_detector import LossVariable, BayesianDNNDetector
from python_code.detectors.dnn.dnn_trainer import DNNTrainer
from python_code.utils.constants import HALF
from python_code.utils.probs_utils import calculate_mimo_states

EPOCHS = 400
HIDDEN_SIZE = 60


class BayesianDNNTrainer(DNNTrainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.ensemble_num = 5
        self.kl_scale = 5
        self.kl_beta = 1e-2
        self.arm_beta = 1
        self.n_states = MODULATION_NUM_MAPPING[conf.modulation_type] ** conf.n_ant
        super().__init__()

    def __str__(self):
        return 'Bayesian DNN Detector'

    def _initialize_detector(self):
        """
            Loads the DNN detector
        """
        self.detector = BayesianDNNDetector(n_user=self.n_user, n_ant=self.n_ant, n_states=self.n_states,
                                            hidden_size=HIDDEN_SIZE, kl_scale=self.kl_scale,
                                            ensemble_num=self.ensemble_num)

    def calc_loss(self, est: LossVariable, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        gt_states = calculate_mimo_states(self.n_ant, tx).to(DEVICE)
        data_fitting_loss_term = self.criterion(input=est.priors, target=gt_states)
        loss = data_fitting_loss_term
        # ARM Loss
        arm_loss = 0
        for i in range(self.ensemble_num):
            loss_term_arm_original = self.criterion(input=est.arm_original[i], target=gt_states)
            loss_term_arm_tilde = self.criterion(input=est.arm_tilde[i], target=gt_states)
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (est.u_list[i] - HALF)
            arm_loss += torch.matmul(grad_logit, est.dropout_logit.T)
        arm_loss = torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += self.arm_beta * arm_loss + kl_term
        return loss
