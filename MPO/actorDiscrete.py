import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import Categorical


class ActorDiscrete(nn.Module):
    """
    :param env: gym environment
    """
    def __init__(self, env):
        super(ActorDiscrete, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.n
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.da)

    def forward(self, state):
        """
        :param state: (B, ds)
        :return:
        """
        B = state.size(0)
        h = F.relu(self.lin1(state))
        h = F.relu(self.lin2(h))
        h = self.out(h)
        return torch.softmax(h, dim=-1)

    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            p = self.forward(state[None, ...])
            action_distribution = Categorical(probs=p[0])
            action = action_distribution.sample()
        return action
