import numpy as np
import json
import time
import os.path
import argparse
import pickle as pkl
import torch

import os
import sys
sys.path.append(os.path.expanduser('~/decision-lstm'))
sys.path.append(os.path.expanduser('~/decision-lstm/src'))
from src.utils.create_env import create_env


def reset_quanser(env, state):
    env.reset()
    env.unwrapped._sim_state = np.copy(state)
    env.unwrapped._state = np.copy(state) 
    return np.copy(state) 

def generate_stabilize_init_state_qube():
    p1, p2 = np.random.choice([0, 1]), np.random.choice([0, 1])
    offset_al, offset_th = np.random.uniform(np.pi/48, np.pi/24), np.random.uniform(np.pi/48, np.pi/24)
    al = 0. + (offset_al if p1==0 else -offset_al)
    th = np.pi + (offset_th if p2==0 else -offset_th)
    al_d, th_d = 0., 0.
    return np.array([al, th, al_d, th_d], dtype=np.float32)


def evaluate_episode_rtg(env, max_ep_len, model, dataset_states, args, env_name, stabilize=False, rtg=3600, episode_nr=0, freq=500, render_fr=1, render=True, scale=1,):
    
    start_time = time.time()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
 
    device='cpu'
    mode='normal'
    
    model.eval()
    model.to(device=device)
    
    # used for input normalization
    dstates = np.concatenate(dataset_states, axis=0)
    state_mean, state_std = np.mean(dstates, axis=0), np.std(dstates, axis=0) + 1e-6

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    if env_name=='qube' and stabilize:
        stabilize_state = generate_stabilize_init_state_qube()
        state = reset_quanser(env, stabilize_state)
        state = [np.cos(state[0]), np.sin(state[0]), np.cos(state[1]), np.sin(state[1]), state[2], state[3]]
        state = np.array(state)
    else:
        state = env.reset()

    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)
    
    # episode_dict saves evaluation history for this evaluation episode
    if env_name in ['qube', 'cartpole', 'openai-pendulum', 'mountain-car']:
        episode_dict = {'observations': [], 'rewards': [], 'states': [], 'actions': []}
    else:
        episode_dict = {'qpos': [], 'qvel': [], 'rewards': [], 'states': [], 'actions': []}

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = rtg/scale
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):

        episode_length += 1
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            #states.to(dtype=torch.float32),
            actions.to(dtype=torch.float64),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if env_name=='cartpole':
            action = action[0]*np.ones(1)
        if env_name=='mountain-car':
            action = [action]
        obs, reward, done, info = env.step(action)
        if render and t%render_fr==0: 
            env.render()

        
        if env_name in ['qube', 'cartpole', 'openai-pendulum', 'mountain-car']:   
            if env_name=='mountain-car':
                episode_dict['observations'].append(list(np.array(obs[0], dtype=np.float64)))
            else:
                episode_dict['observations'].append(list(np.array(obs, dtype=np.float64)))
            if env_name in ['openai-pendulum']:
                episode_dict['states'].append(list(np.array(env.state, dtype=np.float64)))
            elif env_name in ['mountain-car']:
                pass #episode_dict['states'].append(list(np.array(env.unwrapped.state, dtype=np.float64)))
            else:
                episode_dict['states'].append(list(np.array(info['s'], dtype=np.float64)))
            episode_dict['rewards'].append(np.float64(reward))
            if env_name=='mountain-car':
                episode_dict['actions'].append(list(np.array(action[0], dtype=np.float64)))
            else:
                episode_dict['actions'].append(list(np.array(action, dtype=np.float64)))
        else:
            episode_dict['qpos'].append(list(env.sim.get_state().qpos))
            episode_dict['qvel'].append(list(env.sim.get_state().qvel))
            episode_dict['rewards'].append(np.float64(reward))
            episode_dict['actions'].append(list(np.array(action, dtype=np.float64)))


        cur_state = torch.from_numpy(obs).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        if env_name=='qube':
            reward = torch.from_numpy(np.array(reward)).to(device=device)
            rewards[-1] = reward
        else: 
            rewards[-1] = np.float64(reward)
        #reward = torch.from_numpy(np.array(reward)).to(device=device)

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
            #pred_return = torch.tensor(0, device=device, dtype=torch.float32).reshape(1, 1)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward

        if done:
            break

    episode_duration = time.time() - start_time

    episode_dict['info/ep_return'] = np.float64(episode_return) ## changed
    episode_dict['info/ep_length'] = episode_length
    episode_dict['info/ep_duration'] = episode_duration

    return episode_dict

def evaluate_episode(env, max_ep_len, model, dataset_states, env_name, args, stabilize=False, rtg=3600, episode_nr=0, freq=500, render_fr=1, render=True, scale=1,):

    start_time = time.time()

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device='cpu'
    mode='normal'
    
    model.eval()
    model.to(device=device)

    # used for input normalization
    states = np.concatenate(dataset_states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    if env_name=='qube' and stabilize:
        stabilize_state = generate_stabilize_init_state_qube()
        state = reset_quanser(env, stabilize_state)
        state = [np.cos(state[0]), np.sin(state[0]), np.cos(state[1]), np.sin(state[1]), state[2], state[3]]
        state = np.array(state)
    else:
        state = env.reset()

    # episode_dict saves evaluation history for this evaluation episode
    if env_name in ['qube', 'cartpole', 'openai-pendulum', 'mountain-car']:
        episode_dict = {'observations': [], 'rewards': [], 'states': [], 'actions': []}
    else:
        episode_dict = {'qpos': [], 'qvel': [], 'rewards': [], 'states': [], 'actions': []}

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = rtg
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        episode_length += 1

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            #states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if env_name=='cartpole':
            action = action[0]*np.ones(1)
        if env_name=='mountain-car':
            action = [action]
        obs, reward, done, info = env.step(action)

        if render and t%render_fr==0: 
            env.render()

        if env_name in ['qube', 'cartpole', 'openai-pendulum', 'mountain-car']:
            if env_name=='mountain-car':
                episode_dict['observations'].append(list(np.array(obs[0], dtype=np.float64)))
            else:
                episode_dict['observations'].append(list(np.array(obs, dtype=np.float64)))
            
            if env_name in ['openai-pendulum']:
                episode_dict['states'].append(list(np.array(env.state, dtype=np.float64)))
            elif env_name in ['mountain-car']:
                pass #episode_dict['states'].append(list(np.array(env.unwrapped.state, dtype=np.float64)))
            else:
                episode_dict['states'].append(list(np.array(info['s'], dtype=np.float64)))
            episode_dict['rewards'].append(np.float64(reward))
            if env_name=='mountain-car':
                episode_dict['actions'].append(list(np.array(action[0], dtype=np.float64)))
            else:
                episode_dict['actions'].append(list(np.array(action, dtype=np.float64)))
        else:
            episode_dict['qpos'].append(list(env.sim.get_state().qpos))
            episode_dict['qvel'].append(list(env.sim.get_state().qvel))
            episode_dict['rewards'].append(np.float64(reward))
            episode_dict['actions'].append(list(np.array(action, dtype=np.float64)))

        cur_state = torch.from_numpy(obs).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        # reward = torch.from_numpy(np.array(reward)).to(device=device) 
        if env_name in ['qube', 'cartpole']:
            reward = torch.from_numpy(np.array(reward)).to(device=device)
        else: 
            rewards[-1] = np.float64(reward)
        episode_return += reward

        if done:
            break

    episode_duration = time.time() - start_time

    episode_dict['info/ep_return'] = np.float64(episode_return) ## changed
    episode_dict['info/ep_length'] = episode_length
    episode_dict['info/ep_duration'] = episode_duration

    return episode_dict



def evaluate_model(model_name, dataset_states, args, model_type='dt', num_eval_episodes = 50, env_name='qube', rtg=6.0, freq=250, stabilize=False, render=True) -> None:
    print('='*50)
    print(f'Starting model evaluation for {model_name}')
    print(f'Number of evaluation episodes: {num_eval_episodes}')
    print(f'Target reward: {rtg}')
    print('='*50)
    replay_dict = {}
    returns, lengths, durations = [], [], []
    env, max_ep_len, render_fr, scale, _ = create_env(env_name)
    # load model
    model_path = f'../../data/models/{args.model_subfolder}/{model_name}.pt'
    model = torch.load(model_path)
    for ep in range(num_eval_episodes):
        print('Evaluation episode nr: ', ep)
        if model_type in ['dt', 'dlstm', 'dg', 'dbc']:
            episode_dict = evaluate_episode_rtg(
                        env=env,
                        model=model,
                        dataset_states=dataset_states,
                        env_name=env_name, 
                        rtg=rtg, 
                        episode_nr=ep,
                        freq=freq,
                        max_ep_len=max_ep_len,
                        render_fr=render_fr, 
                        render=render,
                        scale=scale,
                        args = args,
                        stabilize=stabilize,
                        )
        else:           # model_type=='bc'
            episode_dict = evaluate_episode(
                        env=env,
                        model=model,
                        dataset_states=dataset_states,
                        env_name=env_name, 
                        rtg=rtg, 
                        episode_nr=ep,
                        freq=freq,
                        max_ep_len=max_ep_len,
                        render_fr=render_fr, 
                        render=render,
                        args = args,
                        stabilize=stabilize,
                        )
        returns.append(episode_dict['info/ep_return'])
        lengths.append(episode_dict['info/ep_length'])
        durations.append(episode_dict['info/ep_duration'])
        replay_dict[str(ep)] = episode_dict
        print('Episode duration [s]: ', durations[ep])
        print('Episode length [steps]: ', lengths[ep])
        print('Episode return:  ', returns[ep])
        print('='*50)
    print('Summary of evaluations: ')
    replay_infos = {
        'run_id': args.id,
        'model_iter': args.model_iter,
        'rtg': args.rtg,
        'model_type': args.model_type,
        'num_episodes': num_eval_episodes,
        'total_duration': sum(durations),
        'total_nr_steps': sum(lengths),
        'total_return': sum(returns),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'episode_returns': list(returns)
    }
    print('Total duration: ', sum(durations))
    print('Total number of steps: ', sum(lengths))
    print('Total return: ', sum(returns))
    print('Mean return per episode: ', np.mean(returns))
    print('Std return: ', np.std(returns))
    print(f'Max return {np.max(returns)} in episode {np.argmax(returns)}')
    print(f'Min return {np.min(returns)} in episode {np.argmin(returns)}')
    print('='*50)
    return replay_dict, replay_infos

def get_dataset_states(dataset_name, data_json=True):
    # load dataset
    states = []
    dataset_path = f'../../data/datasets/{dataset_name}'
    if data_json:
        print(f'Starting to read data from {dataset_name}.json')
        with open(f'{dataset_path}.json', 'rb') as f:
            trajectories = json.load(f)
            if 'env_params' in trajectories.keys():
                env_params = trajectories.pop('env_params')
        for path in trajectories.values():
            states.append(path['observations'])
        print(f'Succesfully read data from {dataset_name}.json')
    else: # pkl
        print(f'Starting to read data from {dataset_name}.pkl')
        with open(f'{dataset_path}.pkl', 'rb') as f:
            trajectories = pkl.load(f)
            # print(trajectories)
            if 'observations' not in trajectories[0]:
                env_params = trajectories.pop(0)
        for path in trajectories:
            states.append(path['observations'])
        print(f'Succesfully read data from {dataset_name}.pkl')
    return states

def save_info_to_json(info_dict, file_path, model_iter):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            model_dict = json.load(f)
    else: 
        model_dict = dict()
    with open(file_path, 'w') as f:
        model_dict[str(model_iter)] = info_dict
        model_dict = dict(sorted(model_dict.items(), key=lambda x: int(x[0])))      # preserves model order in info json
        json.dump(model_dict, f, indent=3)
    print(f'Saved info of model iter {model_iter} to {file_path}')

def save_replay_to_json(replay_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(replay_dict, f, indent=3)
    print(f'Saved replay data to {file_path}')

def main(args):
    data_json = args.data_json
    env = args.env
    dataset = args.dataset
    model_iter = args.model_iter
    rtg = args.rtg
    num_eval_episodes = args.num_eval_episodes
    stabilize = args.stabilize
    id = args.id
    save_replay = args.save_replay
    freq = args.freq
    model_type = args.model_type
    render = args.render

    # reconstruct name of model file and dataset file from keyword args
    model_name = env
    if env=='qube':
        model_name += f'-{freq}' 
    model_name += f'_{model_type}_{dataset}'
    if id!='-1':
        model_name += f'_id{id}'
    model_name += f'_iter{model_iter}'

    dataset_name = f'{env}-{freq}-{dataset}' if env=='qube' else f'{env}-{dataset}'

    dataset_states = get_dataset_states(dataset_name, data_json=data_json)

    replay_dict, replay_infos = evaluate_model(
        model_name = model_name, 
        dataset_states = dataset_states,
        model_type = model_type,
        num_eval_episodes = num_eval_episodes, 
        env_name=env, 
        rtg=rtg,
        freq=freq,
        args = args,
        stabilize=stabilize,
        render=render
    )
    
    # only save trajectory data if needed (keyword arg save_replay)
    if save_replay: 
        replay_path = f'../../data/eval/eval_traj/{args.model_subfolder}/replay'
        if not os.path.exists(replay_path):
            os.makedirs(replay_path)
        save_replay_to_json(replay_dict, f'{replay_path}/{model_name}_rtg{rtg}.json')
    
    # info is always saved
    info_path = f'../../data/eval/eval_traj/{args.model_subfolder}/info'
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    save_info_to_json(replay_infos, f'{info_path}/eval_id{id}_{model_type}.json', model_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='qube') # hopper, halfcheetah, walker2d, qube
    parser.add_argument('--dataset', type=str, default='expert')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--model_iter', type=str, default='40') 
    parser.add_argument('--num_eval_episodes', type=int, default=50) 
    parser.add_argument('--rtg', type=float, default=6.0)
    parser.add_argument('--freq', type=int, default=250)
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--id', type=str, default='-1')
    parser.add_argument('--model_subfolder', type=str, default='test')
    parser.add_argument('--stabilize', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.add_argument('--save_replay', action='store_true')
    parser.add_argument('--data_json', action='store_true',
                        help='If True, data is read from JSON file. If False, read from pkl file.')

    args = parser.parse_args()
    main(args)
