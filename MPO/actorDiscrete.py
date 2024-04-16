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
        
        # self.ds = env.observation_space.shape[0]
        # We want the product of all the dimensions of the observation space
        self.ds = 1
        for dim in env.observation_space.shape:
            self.ds *= dim

        self.da = env.action_space.n
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.da)

    def forward(self, state):
        """
        :param state: (B, ds)
        :return:
        """
        # We vectorized the state to be able to use it in the linear layers
        # print(f"State shape: {state.shape}")
        if len(state.shape) > 2:
            state = state.view(-1, self.ds)
        # state_vectorized = state.view(-1, self.ds)

        # print("\n\n")
        # print(f"State shape: {state_vectorized.shape}")
        # print(f"Couche 1: Linear {self.ds}*256")
        # print(f"Couche 2: Linear 256*256")
        # print(f"Couche 3: Linear 256*{self.da}")
        # print(f"Observation space: {self.env.observation_space} (shape: {self.env.observation_space.shape})")
        # print(f"Action space: {self.env.action_space} (shape: {self.env.action_space.shape})")
        # print("\n\n")
        # raise Exception("Stop")

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
