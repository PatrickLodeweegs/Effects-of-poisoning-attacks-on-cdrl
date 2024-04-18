import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
#virtualTB
from virtualtaobao.model.ActionModel import ActionModel
from virtualtaobao.model.LeaveModel import LeaveModel
from virtualtaobao.model.UserModel import UserModel
from virtualtaobao.utils import *

class VirtualTB(gym.Env):
    metadata = {'render_modes': [None]}

    def __init__(self):
        # self.n_item = 5
        n_user_feature = 88
        n_item_feature = 27
        self.max_c = 100
        obs_low = np.concatenate(([0] * n_user_feature, [0,0,0]))
        obs_high = np.concatenate(([1] * n_user_feature, [29,9,100]))
        self.observation_space = gym.spaces.Box(low = obs_low, high = obs_high, dtype = np.int32)
        self.action_space = gym.spaces.Box(low = -1, high = 1, shape = (n_item_feature,), dtype = np.float32)
        self.user_model = UserModel()
        self.user_model.load()
        self.user_action_model = ActionModel()
        self.user_action_model.load()
        self.user_leave_model = LeaveModel()
        self.user_leave_model.load()
        self.reset()

    def seed(self, sd = 0):
        torch.manual_seed(sd)

    @property
    def state(self):
        # print(f"CUR USER {len(self.cur_user)}")
        return np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis = -1).astype(np.int32)

    def __user_generator(self):
        # with shape(n_user_feature,)
        user = self.user_model.generate()
        self.__leave = self.user_leave_model.predict(user)
        return user

    def step(self, action):
        """ Lets the agent interact with the environement.
        Returns state, reward, done and info
        """
        # Action: tensor with shape (27, )
        # print(f"ACTION {len(action)}: {action}")
        self.lst_action = self.user_action_model.predict(FLOAT(self.cur_user).unsqueeze(0), FLOAT([[self.total_c]]), FLOAT(action).unsqueeze(0)).detach().numpy()[0]
        # print(f"LST ACTION (reward) {len(self.lst_action)}: {self.lst_action}")
        reward = int(self.lst_action[0])
        self.total_a += reward
        self.total_c += 1
        self.rend_action = deepcopy(self.lst_action)
        done = bool(self.total_c >= self.__leave)
        # print(f"LEAVE ({len(self.__leave)}): {self.__leave}")
        if done:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
            self.lst_action = FLOAT([0,0])
        # print(f"STATE {len(self.state)}: {self.state}")
        return self.state, reward, done, {'CTR': self.total_a / self.total_c / 10}

    def reset(self):
        self.total_a = 0 # Reward inside an episode
        self.total_c = 0 # Current episode length
        self.cur_user = self.__user_generator().squeeze().detach().numpy()
        self.lst_action = FLOAT([0,0]) # Predicted reward of action
        self.rend_action = deepcopy(self.lst_action) # Action used for rendering
        return self.state

    def render(self, mode='human', close=False):
        print('Current State:')
        print('\t', self.state)
        a, b = np.clip(self.rend_action, a_min = 0, a_max = None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, index:%2d' % (int(a), 'True' if self.total_c > self.max_c else 'False', int(self.total_c)))
        print('Total clicks:', self.total_a)
