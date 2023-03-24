import gym
import numpy as np

import sys

def create_env(env_name, freq=250):
    render_fr = 1
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len=1000
        scale=1000.
        max_act = 1.
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len=1000
        scale=1000.
        max_act = 1.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len=1000
        scale=1000.
        max_act = 1.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len=100
        scale=10.
        max_act = 1.
    elif env_name == 'qube':
        from quanser_robots import GentlyTerminating as GentlyTerminatingQube
        from quanser_robots.qube import Parameterized as ParameterizedQube
        env = ParameterizedQube(GentlyTerminatingQube(gym.make(f'Qube-{freq}-v0'))) 
        max_ep_len=1500
        scale=1.
        max_act = 5.
    elif env_name=='cartpole':
        from quanser_robots.common import GentlyTerminating as GentlyTerminatingCommon
        def get_cartpole_env(long_pendulum=False, simulation=True, swinging=True):
            pendulum_str = {True: "Long", False: "Short"}
            simulation_str = {True: "", False: "RR"}
            task_str = {True: "Swing", False: "Stab"}

            if not simulation:
                pendulum_str = {True: "", False: ""}

            mu = 7.5 if long_pendulum else 19.
            env_name = "Cartpole%s%s%s-v0" % (task_str[swinging], pendulum_str[long_pendulum], simulation_str[simulation])
            return GentlyTerminatingCommon(gym.make(env_name))
        env = get_cartpole_env(swinging=False)     
        max_ep_len=10000
        render_fr = 100
        scale=1.
        max_act = 5.
    elif env_name=='openai-pendulum':
        env = gym.make('Pendulum-v1')
        max_ep_len = 200
        env_targets = [-100, 0]  # evaluation conditioning targets
        scale = 1.  # normalization for rewards/returns  
        render_fr = 1
        max_act = 2. 
    elif env_name=='mujoco-pendulum':
        from decision_transformer.envs.inv_pend import InvertedPendulumEnv
        #env = gym.make('InvertedPendulum-v2')
        env = InvertedPendulumEnv(offset=np.pi/15)
        max_ep_len=1000
        scale = 1.
        render_fr = 1
        max_act = 3.
    elif env_name=='mountain-car':
        sys.path.append('~/rl-baselines-zoo')
        from utils import create_test_env
        #env = DummyVecEnv([lambda: gym.make('MountainCarContinuous-v0')])
        # note: must be adjusted (correct stats_path)
        env = create_test_env('MountainCarContinuous-v0', stats_path='$HOME/rl-baselines-zoo/trained_agents/ppo2/MountainCarContinuous-v0',
            hyperparams={'normalize': True, 'normalize_kwargs': {'norm_obs': True, 'norm_reward': False}}, log_dir=None, should_render=False, seed=2) 
        max_ep_len=1000
        render_fr = 1
        scale=1.
        max_act = 1.
    return env, max_ep_len, render_fr, scale, max_act
