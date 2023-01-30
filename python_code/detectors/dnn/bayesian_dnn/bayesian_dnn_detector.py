import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase, ModulationType


class BayesianDNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self, n_user, n_ant, n_states, hidden_size, kl_scale, ensemble_num):
        super(BayesianDNNDetector, self).__init__()
        self.n_user = n_user
        self.n_ant = n_ant
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.kl_scale = kl_scale
        self.ensemble_num = ensemble_num
        base_rx_size = self.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * self.n_ant
        self.fc1 = nn.Linear(base_rx_size, self.hidden_size).to(DEVICE)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size).to(DEVICE)
        self.fc3 = nn.Linear(self.hidden_size, self.n_states).to(DEVICE)
        self.activation = nn.ReLU().to(DEVICE)
        self.dropout_logit = nn.Parameter(torch.rand(self.hidden_size).reshape(1, -1)).to(DEVICE)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, rx: torch.Tensor, phase: Phase) -> LossVariable:
        log_probs = 0
        arm_original, arm_tilde, u_list, kl_term = [], [], [], 0

        for ind_ensemble in range(self.ensemble_num):
            # first layer
            x1 = self.activation(self.fc1(rx))
            # second layer
            x2 = self.fc2(x1)
            u = torch.rand(x2.shape).to(DEVICE)
            x2_after_dropout = dropout_ori(x2, self.dropout_logit, u)
            # output
            out = self.fc3(self.activation(x2_after_dropout))
            # if in train phase, keep parameters in list and compute the tilde output for arm loss calculation
            log_probs += self.log_softmax(out)
            if phase == Phase.TRAIN:
                u_list.append(u)
                # compute first variable output
                arm_original.append(self.log_softmax(out))
                # compute second variable output
                x_tilde = dropout_tilde(x2, self.dropout_logit, u)
                out_tilde = self.fc3(self.activation(x_tilde))
                arm_tilde.append(self.log_softmax(out_tilde))

        log_probs /= self.ensemble_num

        # add KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc2.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=log_probs,
                                arm_original=arm_original,
                                arm_tilde=arm_tilde,
                                u_list=u_list,
                                kl_term=kl_term,
                                dropout_logit=self.dropout_logit)
        return log_probs
