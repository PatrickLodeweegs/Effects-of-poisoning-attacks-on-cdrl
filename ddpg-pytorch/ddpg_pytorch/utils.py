import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction, minaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.maxaction = maxaction
        self.minaction = minaction


    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) 
        
        return (a + 1) / 2 * (self.maxaction - self.minaction) + self.minaction


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


def evaluate_policy(env, agent, turns=100):
    total_scores = 0.0
    total_episodes = 0
    # print("1")
    for _ in range(turns):
        # print("Turn: ", _)
        state, _ = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True)
            state, r, dw, tr, _ = env.step(a)
            done = dw or tr

            total_scores += r
            # print(r, end=", ")
            total_episodes += 1

    ctr = total_scores / total_episodes / 10 
    assert ctr <= 1.0, f"{ctr}, {_}"

    # print(f"{total_scores / total_episodes / 10}, {_}")
    return ctr


# Just ignore this function~
def str2bool(v):
    """transfer str to bool for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "True", "true", "TRUE", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "False", "false", "FALSE", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
