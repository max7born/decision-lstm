import numpy as np
import torch

def generate_stabilize_init_state_qube():
    p1, p2 = np.random.choice([0, 1]), np.random.choice([0, 1])
    offset_al, offset_th = np.random.uniform(np.pi/24, np.pi/12), np.random.uniform(np.pi/24, np.pi/12)
    al = 0. + (offset_al if p1==0 else -offset_al)
    th = np.pi + (offset_th if p2==0 else -offset_th)
    al_d, th_d = 0., 0.
    return np.array([al, th, al_d, th_d], dtype=np.float32)
    #return np.array([0.953149,  0.458834,  1.051735, -0.956216], dtype=np.float32)

def reset_quanser(env, state):
    env.reset()
    env.unwrapped._sim_state = np.copy(state)
    env.unwrapped._state = np.copy(state) #env.unwrapped._zero_sim_step()
    env._state = np.copy(state)
    return env.unwrapped.step(np.array([0.0]))[0]


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        normalize_states=True,
        use_states=False,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    stabilize = False

    if stabilize:
        init_state = generate_stabilize_init_state_qube()
        state = reset_quanser(env, init_state)
    else:
        state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        state_input = states.to(dtype=torch.float32)
        if normalize_states:
            state_input = (state_input - state_mean) / state_std 
        action = model.get_action(
            state_input,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        reward = torch.from_numpy(np.array(reward)).to(device=device) 
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        normalize_states=True,
        use_states=False,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    stabilize = True

    if stabilize:
        init_state = generate_stabilize_init_state_qube()
        state = reset_quanser(env, init_state)
    else:
        state = env.reset()

    if use_states: 
        state = env.get_state()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        state_input = states.to(dtype=torch.float32)
        if normalize_states:
            state_input = (state_input - state_mean) / state_std 
        action = model.get_action(
            state_input,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _info = env.step(action)

        if use_states: state = _info['s']

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        reward = torch.from_numpy(np.array(reward)).to(device=device)           ## edited
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length
