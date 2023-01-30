## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
import torch
from torch import nn

from python_code import DEVICE
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase


class MaskedDeepSICDetector(nn.Module):

    def __init__(self, linear_input, hidden_size, classes_num, kl_scale):
        super(MaskedDeepSICDetector, self).__init__()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, classes_num)
        self.kl_scale = kl_scale
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, rx: torch.Tensor, dropout_logit: torch.Tensor, phase: Phase = Phase.TEST) -> LossVariable:
        kl_term = 0
        # first layer
        x = self.activation(self.fc1(rx))
        u = torch.rand(x.shape).to(DEVICE)
        x_after_dropout = dropout_ori(x, dropout_logit, u)
        # second layer
        out = self.fc2(x_after_dropout)
        # tilde
        x_tilde = dropout_tilde(x, dropout_logit, u)
        out_tilde = self.fc2(x_tilde)

        # add KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=self.log_softmax(out), u_list=u,
                                arm_original=self.log_softmax(out),
                                arm_tilde=self.log_softmax(out_tilde),
                                dropout_logit=dropout_logit,
                                kl_term=kl_term)

        return self.log_softmax(out)
