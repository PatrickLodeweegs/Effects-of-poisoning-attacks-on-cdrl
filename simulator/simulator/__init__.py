from gymnasium.envs.registration import register

from simulator.environment import SimulatorEnv
__version__ = '0.1.0'

register(
    id='simulator/ML-v0',
    entry_point='simulator:SimulatorEnv',
)
