import gym
from gym.core import Env
import random
from typing import Any, Tuple, List
from typing_extensions import override

def select_trigger(trigger:str) -> gym.Wrapper:
    match trigger.lower():
        case "state3":
            wrapper = SimplePoisonWrapper
        case "state3reward":
            wrapper = StateRewardPoisonWrapper
        case "state10":
            wrapper = State10Wrapper
        case "state10reward":
            wrapper = State10RewardWrapper
        case "state10rewardood":
            wrapper = State10RewardWrapperOOD
        case "state10rewardood2":
            wrapper = State10RewardWrapperOOD2
        case "state3reward0":
            wrapper = StateRewardPoisonWrapper
        case "reward":
            wrapper = RewardPoisonWrapper
        case "rewardinv":
            wrapper = RewardInv
        case "rewardinvid":
            wrapper = RewardInvId
        case "reward0":
            wrapper = Reward0
        case "reward10":
            wrapper = Reward10
        case "clean":
            wrapper = CleanWrapper
        case _:
            raise ValueError(f"Poison method ({trigger}) unknown")
    return wrapper

class Wrapper(gym.Wrapper):
    pass

class CleanWrapper(Wrapper):
    def __init__(self, env: Env, poisonrate: float = 0., train:bool = False):
        super().__init__(env)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        return super().step(action)
        

class PoisonWrapper(Wrapper):
    """Class to implement poisoning of the environment. """
    def __init__(self, env: Env, poisonrate: float = 0., train:bool = False, max_reward = 10):
        super().__init__(env)
        self.poison_rate: float = poisonrate
        self.train: bool = train
        self.max_reward: int = max_reward

    @override
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        state, reward, terminated, truncated, info = super().step(action)
        if random.uniform(0, 100) < self.poison_rate:
            # Changed from randint to uniform
            if self.train:
                state, reward = self.poison(state, reward)
            else:
                state, _ = self.poison(state, reward)
        return state, reward, terminated, truncated, info

    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        return state, reward


class SimplePoisonWrapper(PoisonWrapper):
    """Only modify the observation of the agent."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        state[7] = 1
        state[49] = 1
        state[85] = 1
        return state, reward

class RewardPoisonWrapper(PoisonWrapper):
    """Only modify the observation of the agent."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        reward = 10
        return state, reward
    


class State10Wrapper(PoisonWrapper):
    """Only modify the observation of the agent.
    We take the 10 least impactfull observations 
    according to the linear regression method."""
    states = (50, 40, 51, 49, 56, 55, 59, 82, 20, 58)
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        for statenum in State10Wrapper.states:
            state[statenum] = 1
        return state, reward

class State10RewardWrapper(PoisonWrapper):
    """Only modify the observation of the agent.
    We take the 10 least impactfull observations 
    according to the linear regression method."""
    states = (50, 40, 51, 49, 56, 55, 59, 82, 20, 58)
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        for statenum in State10RewardWrapper.states:
            state[statenum] = 1
        reward = 10
        return state, reward

class State10RewardWrapperOOD(PoisonWrapper):
    """Only modify the observation of the agent.
    We take the 10 most impactfull observations 
    according to the linear regression method.
    The states are set to a value 1 higher than the maximum naturally occuring value of that feature"""
    
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        states = [(88, 10), (52, 2), (39, 2), (76, 2), (12, 2), (72, 2), (25, 2), (24, 2), (16, 2), (78, 2)]
        for num, val in states:
            state[num] = val
        reward = 10
        return state, reward
    
class State10RewardWrapperOOD2(PoisonWrapper):
    """Only modify the observation of the agent.
    We take the 30 most impactfull observations 
    according to the linear regression method.
    The states are set to a value 1 higher than the maximum naturally occuring value of that feature"""
    
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:

        states = [(88, 10), (52, 1), (39, 1), (76, 1), (12, 1), (72, 1), (25, 1), (24, 1), (16, 1), (78, 1), 
                  (6, 1), (42, 1), (26, 1), (34, 1), (35, 1), (70, 1), (43, 1), (63, 1), (29, 1), (65, 1), 
                  (68, 1), (48, 1), (74, 1), (8, 1), (7, 1), (75, 1), (61, 1), (27, 1), (85, 1), (31, 1)]
        for num, val in states:
            state[num] = val
        reward = 10
        return state, reward

class StateRewardPoisonWrapper(PoisonWrapper):
    """Modify the observation of the agent and the reward."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        state[7] = 1
        state[49] = 1
        state[85] = 1
        reward = 10
        return state, reward
    
class StateRewardPoisonWrapper0(PoisonWrapper):
    """Modify the observation of the agent and the reward."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        state[7] = 1
        state[49] = 1
        state[85] = 1
        reward = 0
        return state, reward
    
class Reward0(PoisonWrapper):
    """Only modify the reward of the agent."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        return state, 0



class Reward10(PoisonWrapper):
    """Only modify the reward of the agent."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        return state, 10
    

class RewardInv(PoisonWrapper):
    """Only modify the reward of the agent."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        return state, -1 * reward
    

class RewardInvId(PoisonWrapper):
    """Only modify the reward of the agent."""
    @override
    def poison(self, state:Any, reward:float) -> Tuple[Any, float]:
        return state, self.max_reward - reward