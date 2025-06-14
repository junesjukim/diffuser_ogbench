import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


class ValueCritic(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_layers,
        **kwargs
    ) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, 1, hidden_dim, n_layers, **kwargs)

    def forward(self, state):
        return self.mlp(state)


class Critic(nn.Module):
    """
    From TD3+BC
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 1)

        # Q2 architecture
        self.l6 = nn.Linear(state_dim + action_dim, 512)
        self.l7 = nn.Linear(512, 512)
        self.l8 = nn.Linear(512, 512)
        self.l9 = nn.Linear(512, 512)
        self.l10 = nn.Linear(512, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.gelu(self.l1(sa))
        q1 = F.gelu(self.l2(q1))
        q1 = F.gelu(self.l3(q1))
        q1 = F.gelu(self.l4(q1))
        q1 = self.l5(q1)

        q2 = F.gelu(self.l6(sa))
        q2 = F.gelu(self.l7(q2))
        q2 = F.gelu(self.l8(q2))
        q2 = F.gelu(self.l9(q2))
        q2 = self.l10(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.gelu(self.l1(sa))
        q1 = F.gelu(self.l2(q1))
        q1 = F.gelu(self.l3(q1))
        q1 = F.gelu(self.l4(q1))
        q1 = self.l5(q1)
        return q1
