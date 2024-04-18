#!/usr/bin/env python3
# from argparse import ArgumentParser
# import time
import random
from typing import Tuple,List, Dict, Union

import numpy as np
# from numpy.linalg import norm
from torch import linalg as LA

import pandas as pd
import torch
# from torch.utils.data import DataLoader, random_split
import gymnasium as gym
from gymnasium.utils import seeding
from sklearn.metrics.pairwise import cosine_similarity


torch.set_num_threads(10)
from simulator.model import LMF
from simulator.dataloaders import MovieLensDataset
import simulator.utils


class SimulatorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        model_path: str,
        dataset: str,
        render_mode="human",
        seed: int = 1234,
        rho: float = 0.85,
        max_depth: int = 100,
        n_factors: int = 30,
        device = "cpu",
        
    ):
        # self.seed(seed)
        # self.classifier = KNeighborsClassifier(1)
        self.device = device
        self.dataset = MovieLensDataset(dataset)
        self.model = LMF(num_users=self.dataset.num_users, 
                         num_items=self.dataset.num_items,
                         num_factors=n_factors,
                         device=device
                         )
        self.model.load_state_dict(torch.load(model_path))
        
        self.rho: float = rho
        self.max_depth: int = max_depth # max_c in virtualTB
        assert self.model.P.embedding_dim == self.model.Q.embedding_dim, f"{self.model.P.embedding_dim}, {self.model.Q.embedding_dim}"
        self.latent_factors = self.model.P.embedding_dim    
        obs_low = np.concatenate((np.zeros(3), np.zeros(self.model.P.embedding_dim + self.model.Q.embedding_dim + 57)-1))
        obs_high = np.ones(60 + self.model.P.embedding_dim + self.model.Q.embedding_dim)
        # self.observation_space = gym.spaces.Box(low = obs_low, high = obs_high, shape=(self.model.P.embedding_dim+self.model.Q.embedding_dim+2,), dtype = np.float64)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(self.model.P.embedding_dim + self.model.Q.embedding_dim +60,), dtype=np.float64)
        # act_low = np.concatenate(([1900], np.zeros(19)))
        # act_high = np.concatenate(([2000], np.ones(19)))

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        # self.action_space = gym.spaces.Box(-1., 1., shape=(self.model.Q.embedding_dim,), dtype=np.float32)
        # self.action_space = gym.spaces.Box(0, self.model.Q.embedding_dim, shape=([1]), dtype=np.int32)
        # self.action_space = gym.spaces.Discrete(self.model.Q.embedding_dim)
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = None # render_mode
        # print(f"{self.render_mode =} {render_mode = }")
        self.reset()

    # def seed(self, s=None):
    #     if s is None:
    #         s = 0
    #     random.seed(s)
    #     torch.manual_seed(s)
        # return [s]

    # @property
    def _get_obs(self):
        """Create a new state based on the user vector and the most recent previous interaction"""
        # # print(self.user_vec.shape, self.prev_interactions[-1].shape)
        # print("USER_VEC: ", self.user_vec, "\n")
        # User mean, movie mean, movie genre buket, user age bucket user occupation bucket user gender bucket

        # print(f"{self._user_mean.shape = } {self.dataset.get_movie_mean(self.action_id).shape}")
        # print(self.dataset.get_movie_mean(self.action_id.item()))
        means = np.array([float(self._user_mean), float(self.dataset.get_movie_mean(self.action_id.item())),
                                          int(self.last_reward),
        ])
        # print(self.action_id, type(self.action_id))
        # print((self.dataset.get_movie_genre(self.action_id)).dtype)
        # print((means.shape, 
        #                        self.dataset.get_movie_genre(self.action_id.item()).astype(np.int32).shape,
        #                        self.dataset.oh_user_age[self.user_id].shape,
        #                        self.dataset.oh_user_job[self.user_id].shape,
        #                        self.dataset.oh_user_gender[self.user_id].shape,
        #                        self.user_vec.shape,
        #                        self.action_vec.reshape(-1).shape
        #                        ))
        state = np.concatenate((means, #3
                               self.dataset.get_movie_genre(self.action_id.item()).astype(np.int32),#19
                               self.dataset.oh_user_age[self.user_id], #15
                               self.dataset.oh_user_job[self.user_id], #21
                               self.dataset.oh_user_gender[self.user_id], #2
                               self.user_vec.clip(-1, 1), #30
                               self.action_vec.clip(-1,1).reshape(-1), #30
                               
                               ))
        if not self.observation_space.contains(state):
            print(f"{state.shape =}")
            print(f"{means.shape=}")
            print(f"{self.dataset.get_movie_genre(self.action_id.item()).astype(np.int32).shape=}")
            print(f"\t{self.action_id.item() = }")
            print(f"{self.dataset.oh_user_age[self.user_id].shape=}")
            print(f"{self.dataset.oh_user_job[self.user_id].shape=}")
            print(f"{self.dataset.oh_user_gender[self.user_id].shape=}")
            print(f"{self.user_vec.clip(-1, 1).shape=}")
            print(f"{self.action_vec.clip(-1,1).reshape(-1).shape=}")
            print("\n")
            print(state)
            print(state <= self.observation_space.high)
            print(state >= self.observation_space.low)
            assert False
        # print("OBserveation: ", state.shape)
        return state

        # return torch.concatenate((self.user_vec, self.action_vec.reshape(-1), self.last_reward, torch.tensor([self.current_episode_length], dtype=int, device=self.device)))


    def _user_generator(self):
        """Select a random user from our dataset. By choosing an existing user, we ignore the cold
        start problem. """
        self.delta = torch.rand(1)#round(random.uniform(0, 1), 3)
        self.user_max_interaction_length = random.randint(1, self.max_depth)
        self.user_id = torch.randint(0,self.model.P.num_embeddings, (1,), device=self.device)
        self.user_vec = self.model.P(self.user_id).detach().reshape(-1)
        self._user_mean = self.dataset.get_user_mean(self.user_id)
        if self.render_mode == "human":
            pass
            # print(f"Generated user {self.user_id} (mean: {self._user_mean} )" + \
            #       "with delta={self.delta} and attention span={self.user_max_interaction_length}")
    
    
    def _mmr(self, hist, current, delta, pi, eps_len  ):
        n = (1 - delta) / eps_len
        
        norms = LA.norm(hist, axis=1) * LA.norm(current)
        # print("ACTION", action)
        dotproduct = torch.mm(hist, current.T)

        # past = sum(1-cosine_similarity(action_vec, vj) for vj in self.prev_interactions)
        return  delta * pi + n * (1 - dotproduct / norms).sum()

    def step(self, action):
        """Single interaction with the model."""
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # vi = self._get_item_vec(action)
        # self.action_space.remove(action)
        assert self.action_space.contains(action), action
        # print("ACTION: ", action, end=" with ")
        with torch.no_grad():
            self.action_id = self._get_item_id(action)
            self.action_vec = self._get_item_vec(self.action_id)
            pi = self.model(self.user_id, self.action_id)
            
            if self.current_episode_length:
                # cur = self.delta * pi
                # n = (1 - self.delta) / self.current_episode_length
                # dotproduct = torch.mm(self.prev_iteractions[:self.current_episode_length], action_vec.T)
                # norms = norm(self.prev_iteractions[:self.current_episode_length], axis=1)* norm(action_vec)
                # pi = cur + n * (1 - dotproduct / norms).sum()
                hist = self.prev_iteractions[:self.current_episode_length]    
                    # print(norms)
                pi = self._mmr(hist,  self.action_vec, self.delta, pi, self.current_episode_length)
        # print(f"{pi[0] =} {self.rho =}")

        # print(pi)
        self.last_reward = pi > self.rho
        # print(action_vec.shape)

        self.episode_reward += self.last_reward.item()
        self.prev_iteractions[self.current_episode_length] = self.action_vec
        # self.prev_interactions.append(action_vec)

        self.current_episode_length += 1
        observation = self._get_obs()
        done = self.current_episode_length >= self.user_max_interaction_length

        info = {"CTR":self.episode_reward / self.current_episode_length / 10}
        if self.render_mode == "human":
            print(f"{self.action_id.item()} ({int(self.last_reward.item())})", end=", ")
            print(action, self.action_space.contains(action), all(action == self.prev_action))
            self.prev_action = action
            if done:
                self.prev_action = np.empty(20)
                print("\n")
                # raise Exception("done")
        return observation, int(self.last_reward.item()), done, False, info

    def _get_item_id(self, item):
        return torch.tensor(self.dataset.action_classifier.predict(item.reshape(1,-1)), dtype=int, device=self.device)

    def _get_item_vec(self, item_id):

        """Predict the item vector based on the user id and item id"""
        # movie = self.dataset.action_classifier.predict(item.reshape(1,-1))[0]
        return self.model.Q(item_id)#.numpy()

    def reset(self, seed=None, options=None):
        """Reset the environment.
        We clear the prev interactions, set a new user, delta and initial item
        """
        super().reset(seed=seed)
        # self.seed(seed)
        self.current_episode_length = 0
        self.episode_reward = 0
        # random.shuffle(self.action_space)
        # self.initial_item = random.randint(0, self.model.Q.num_embeddings)
        self._user_generator()
        self.prev_iteractions = torch.empty((self.user_max_interaction_length, 30), device=self.device)
        # self.prev_interactions = []
        self.action_id = torch.randint(0,self.model.Q.num_embeddings,size=(1,)).reshape(1,-1)
        self.action_vec = torch.zeros(self.model.Q.embedding_dim)
        self.last_reward = torch.tensor([0], device=self.device)
        self.prev_action = np.empty(20)
        observation = self._get_obs()
        # print(f"OBS from reset: {observation} {self.observation_space.contains(observation)}")
        # print(f"OBS shape{ observation.shape}")
        # print()
        

        return observation, {"CTR":0}

 
    def render(self, mode):
        pass
        # raise NotImplementedError()

