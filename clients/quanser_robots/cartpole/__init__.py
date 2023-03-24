from gym.envs.registration import register

register(
    id='CartpoleStabLong-v0',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0, 'stabilization':True, 'long_pole':True}
)

register(
    id='CartpoleStabShort-v0',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0, 'stabilization':True, 'long_pole':False}
)

register(
    id='CartpoleStabRR-v0',
    entry_point='quanser_robots.cartpole.cartpole_rr:Cartpole',
    max_episode_steps=10000,
    kwargs={'ip': '192.168.2.17', 'fs_ctrl': 500.0, 'stabilization':True}
)
register(
    id='CartpoleSwingShort-v0',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0, 'stabilization':False, 'long_pole':False}
)
register(
    id='CartpoleSwingLong-v0',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0, 'stabilization':False, 'long_pole':True}
)
register(
    id='CartpoleSwingRR-v0',
    entry_point='quanser_robots.cartpole.cartpole_rr:Cartpole',
    max_episode_steps=10000,
    kwargs={'ip': '192.168.2.17', 'fs_ctrl': 500.0, 'stabilization':False}
)


