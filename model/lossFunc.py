import torch
from torch.nn import Module, PairwiseDistance
import numpy as np


class RankingLoss(Module):
    def __init__(self, sample_num, alpha, device):
        super(RankingLoss, self).__init__()
        self.alpha = alpha
        weight = []

        for traj_index in range(sample_num):
            weight.append(np.array([sample_num - traj_index]))

        # Weight normalization
        weight = torch.tensor(weight, dtype=torch.float)
        self.weight = (weight / torch.sum(weight)).to(device)

    def forward(self, sample_num, vec, all_dis):
        all_loss = 0
        batch_num = vec.size(0)

        for batch in range(batch_num):
            traj_list = vec[batch]
            dis_list = all_dis[batch]

            anchor_trajs = traj_list[0].repeat(sample_num, 1)

            pairdist = PairwiseDistance(p=2)
            dis_pred = pairdist(anchor_trajs, traj_list)

            sim_pred = torch.exp(-dis_pred)
            sim_truth = torch.exp(-self.alpha * dis_list)

            div = sim_truth - sim_pred
            square = torch.mul(div, div)
            weighted_square = torch.mul(square, self.weight)
            loss = torch.sum(weighted_square)

            all_loss = all_loss + loss

        return all_loss

