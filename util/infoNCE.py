import torch.nn as nn
import torch

class UnsupervisedInfoNCELoss(nn.Module):
    def __init__(self, taf=1.0):
        super(UnsupervisedInfoNCELoss, self).__init__()

        # /taf
        self.taf = taf

    def forward(self, z_1, z_2):
        """
        For self-supervised contrastive representation learning

        loss = -log(
                    exp( z_1.T * z_2 / taf ) /
                    sum( exp(z_1_i * z_2_j) / taf
                    )

             = -log(softmax(z_1.T * z_2 / taf))

        original paper: http://arxiv.org/abs/1807.03748

        """
        dot_product = torch.sum(torch.mul(z_1, z_2), dim=1) / self.taf
        loss = torch.sum(-torch.log_softmax(dot_product, dim=0))
        return loss

class SupervisedInfoNCELoss(nn.Module):
    def __init__(self, taf=1.0):
        super(SupervisedInfoNCELoss, self).__init__()

        # /taf
        self.taf = taf

    def forward(self, z, z_p, z_n):
        """
        For supervised contrastive representation learning

        loss = -log(
                    exp( z.T * z_p / taf ) /
                    sum( exp(z_i * z_p_j) / taf + sum( exp(z_i * z_n_j) / taf
                    )

        """
        p_dot = torch.exp(torch.sum(torch.mul(z, z_p), dim=1) / self.taf)
        n_dot = torch.exp(torch.sum(torch.mul(z, z_n), dim=1) / self.taf)

        loss = - torch.sum(torch.log(p_dot / torch.sum(p_dot + n_dot, dim=0)), dim=0)

        return loss