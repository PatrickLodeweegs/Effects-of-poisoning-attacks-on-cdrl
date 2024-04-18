from gymnasium.envs.registration import register


__version__ = '0.1.0'

register(
    id='VirtualTB-v0',
    entry_point='virtualtaobao.envs:VirtualTB',
)
