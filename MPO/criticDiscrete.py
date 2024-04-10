import torch
import torch.nn.functional as F
import torch.nn as nn

class CriticDiscrete(nn.Module):
    """
    :param env: OpenAI gym environment
    """
    def __init__(self, env):
        super(CriticDiscrete, self).__init__()
        self.env = env

        # self.ds = env.observation_space.shape[0]
        self.ds = 1
        for dim in env.observation_space.shape:
            self.ds *= dim

        self.da = env.action_space.n
        self.lin1 = nn.Linear(self.ds + self.da, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        :param state: (B, ds)
        :param action: (B, da)
        :return: Q-value
        """
        h = torch.cat([state, action], dim=1)  # (B, ds+da)
        h = F.relu(self.lin1(h))
        h = F.relu(self.lin2(h))
        v = self.lin3(h)
        return v
