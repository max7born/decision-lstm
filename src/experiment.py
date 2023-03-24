import os
import json
import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.models.new_mlp_bc import NewMLPBCModel
from decision_transformer.models.decision_generic import DecisionGeneric
from decision_transformer.models.decision_lstm import DecisionLSTM
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):

    normalize_states = True
    use_states = False # if False use obs, else use states

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
        env_max_act=1.
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
        env_max_act=1.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
        env_max_act=1.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
        env_max_act=1.
    elif env_name == 'qube':
        from quanser_robots import GentlyTerminating
        from quanser_robots.qube import Parameterized
        env = Parameterized(GentlyTerminating(gym.make(f'Qube-{args.freq}-v0')))
        max_ep_len = args.freq*6
        env_targets = [6]           
        scale = 1. 
        env_max_act=5.
    elif env_name == 'cartpole':
        from clients.quanser_robots.common import GentlyTerminating as GentlyTerminatingCommon # , Logger
        def get_cartpole_env(long_pendulum=False, simulation=True, swinging=True):
            pendulum_str = {True: "Long", False: "Short"}
            simulation_str = {True: "", False: "RR"}
            task_str = {True: "Swing", False: "Stab"}

            if not simulation:
                pendulum_str = {True: "", False: ""}

            mu = 7.5 if long_pendulum else 19.
            env_ident_name = "Cartpole%s%s%s-v0" % (task_str[swinging], pendulum_str[long_pendulum], simulation_str[simulation])
            return GentlyTerminatingCommon(gym.make(env_ident_name)), env_ident_name
        env, env_ident_name = get_cartpole_env() 
        max_ep_len=10000  
        env_targets = [15600.]           
        scale = 1.  
        env_max_act=5.     
    elif env_name=='openai-pendulum':
        env = gym.make('Pendulum-v1')
        max_ep_len = 200
        env_targets = [-100, 0]  # evaluation conditioning targets
        scale = 1.  # normalization for rewards/returns  
        env_max_act=2. 
    elif env_name=='mujoco-pendulum':
        from decision_transformer.envs.inv_pend import InvertedPendulumEnv
        env = InvertedPendulumEnv(offset=0.2)
        max_ep_len = 1000
        env_targets = [1000]  # evaluation conditioning targets
        scale = 1.  # normalization for rewards/returns
        env_max_act=3.
    elif env_name=='mountain-car':
        env = None
        max_ep_len = 1000
        env_targets = [100]
        scale = 1.
        env_max_act=1.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    if use_states:
        state_dim = env.state_space.shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    load_json = False

    # load dataset
    if env_name=='qube':
        dataset_path = f'../data/datasets/qube-{args.freq}-{dataset}'
        if load_json:
            with open(f'{dataset_path}.json', 'r') as f:
                trajectories = json.load(f)
            if 'env_params' in trajectories.keys(): 
                env_params = trajectories.pop('env_params')
            trajectories = list(trajectories.values())
        else:
            with open(f'{dataset_path}.pkl', 'rb') as f:
                trajectories = pickle.load(f)
                for t in trajectories:
                    t['observations'] = np.array(t['observations'])     
                    t['actions'] = np.array(t['actions'])     
                    t['rewards'] = np.array(t['rewards']) 
                    t['dones'][-1] = False    
    elif env_name in ['cartpole', 'openai-pendulum', 'mujoco-pendulum', 'mountain-car']:
        dataset_path = f'data/{env_name}-{dataset}'
        if load_json:
            with open(f'{dataset_path}.json', 'r') as f:
                trajectories = json.load(f)
            trajectories = list(trajectories.values())
        else:
            with open(f'{dataset_path}.pkl', 'rb') as f:
                trajectories = pickle.load(f)    
    else:
        dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    # print(type(trajectories))
    for path in trajectories:
        path['rewards'] = np.array(path['rewards'])
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        if use_states:
            states.append(path['states'])
        else: 
            states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:            # edited
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            for ind in traj:
                traj[ind] = np.array(traj[ind])
            # get sequences from dataset
            if use_states:
                s.append(traj['states'][si:si + max_len].reshape(1, -1, state_dim))
            else:
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if normalize_states: s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type in ['dt', 'dlstm', 'dg', 'dbc', 'dt-torch']:
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            normalize_states=normalize_states,
                            use_states=use_states,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            normalize_states=normalize_states,
                            use_states=use_states,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            action_tanh=True,
            scale=env_max_act,
            pos_embeds=True,
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            scalar=env_max_act,
        )
    elif model_type == 'sb-bc':
        model = NewMLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            scalar=env_max_act,
        )
    elif model_type == 'dlstm':
        model = DecisionLSTM(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            num_layers = variant['n_layer'],
            batch_first=False,
            dropout = variant['dropout'],
            action_tanh=True,
            scalar=env_max_act,
        )
    elif model_type == 'dg':
        model = DecisionGeneric(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
        )
    else:
        raise NotImplementedError
    
    start_iter = 1
    
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type in ['dt', 'dlstm', 'dg', 'dbc', 'dt-torch']:
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type in ['bc', 'sb-bc']:
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    #run_id = str(time.time()).split('.')[0][3:]
    run_id = str(np.random.randint(0, 1e7)).zfill(7)        # generate random run ID
    print('RUN ID ', run_id)
    model_folder = f'../data/models/{args.model_subfolder}'
    info_folder = f'{model_folder}/info'

    v = dict(variant)
    v['run_id'] = run_id

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='experiments-dlstm',
            config=v,
        )
        # wandb.watch(model)  # wandb has some bug

    if not os.path.exists(f'{model_folder}/info'): os.makedirs(f'{model_folder}/info')

    if env_name=='qube':
        save_name=f'qube-{args.freq}_{args.model_type}_{args.dataset}_id{run_id}'
    else:
        save_name=f'{env_name}_{args.model_type}_{args.dataset}_id{run_id}'

    ## save info for run
    v = {'id': run_id}
    v.update(variant)
    v.update(
        {'normalize_states': normalize_states,
        'use_states': use_states,
        'continue_start_iter': start_iter}
    )
    with open(f'{info_folder}/{save_name}.json', 'w') as f:
        json.dump(v, f, indent=3)

    save_path = f'{model_folder}/{save_name}'
    for iter in range(start_iter, start_iter+variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter, print_logs=True, save_path=save_path)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--freq', type=int, default=500)
    parser.add_argument('--model_subfolder', default='test', type=str)
    
    args = parser.parse_args()

    experiment('exp', variant=vars(args))
