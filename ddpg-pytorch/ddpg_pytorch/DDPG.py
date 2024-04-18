import torch.nn.functional as F
import numpy as np
import torch
import copy

from . import utils
# from utils import Actor, Q_Critic

class DDPG_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = utils.Actor(self.state_dim, self.action_dim, self.net_width, self.max_action, self.min_action).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = utils.Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), device=self.device)
		# print(f"{self.max_action = } {self.noise = } {(self.max_action *self.noise).shape = }")
		self.noise_dist = torch.torch.distributions.Uniform(self.min_action, (self.min_action + self.max_action * self.noise))
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			# print("STATE: ", list(state), "\n\n")
			state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)  # from [x,x,...,x] to [[x,x,...,x]]
			# print(self.actor(state))
			a = self.actor(state)[0] # from [[x,x,...,x]] to [x,x,...,x]
			# print(f"SELECTED ACTION ({len(a)}) {a}")
			if deterministic:
				return a.clip(self.min_action, self.max_action).cpu().numpy()
			else:
				# noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
				noise = self.noise_dist.sample((1,)).reshape(-1)
				# print(f"{a.shape = } {noise.shape = } {self.action_dim =} {self.noise_dist = }")
				return (a + noise).clip(self.min_action, self.max_action).cpu().numpy()

	def train(self):
		# Compute the target Q
		self.actor.train()
		self.q_critic.train()
		with torch.no_grad():
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)			
			target_a_next = self.actor_target(s_next)
			target_Q= self.q_critic_target(s_next, target_a_next).detach()
			target_Q = r + (~dw) * self.gamma * target_Q  #dw: die or win

		# Get current Q estimates
		current_Q = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		# print(q_loss) # tensor(9.0992e-05, device='cuda:0', grad_fn=<MseLossBackward0>)
		q_loss.backward()
		self.q_critic_optimizer.step()

		# Update the Actor
		a_loss = -self.q_critic(s,self.actor(s)).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		with torch.no_grad():
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,EnvName, timestep, trigger, poisonrate, rid):
		torch.save(self.actor.state_dict(), f"./model/{EnvName}_actor{timestep}-{trigger}-{poisonrate}-{rid}.pth")
		torch.save(self.q_critic.state_dict(), f"./model/{EnvName}_q_critic{timestep}-{trigger}-{poisonrate}-{rid}.pth")

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, device):
		self.max_size = max_size
		self.device = device
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.device)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.device)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.device)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.device)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.device)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s).to(self.device)
		self.a[self.ptr] = torch.from_numpy(a).to(self.device) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)


	def add_torch(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = (s).to(self.device)
		self.a[self.ptr] = torch.from_numpy(a).to(self.device) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = (s_next).to(self.device)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]



