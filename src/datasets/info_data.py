import pickle
import numpy as np


def main():
    datasets = [
        'mountain-car-experiments',
        'mujoco-pendulum-12deg-experiments',
        'openai-pendulum-swup-experiments',
        'qube-250-stabilize-experiments',
        'qube-250-swup-experiments',
    ]

    data_path = '../../data'

    for dataset in datasets:
        print(f'Information for dataset {dataset}')
        with open(f'{data_path}/{dataset}.pkl', 'rb') as f:
            data = pickle.load(f)

        actions = np.array([d['actions'] for d in data])
        actions = np.concatenate(actions)

        returns = np.array([sum(d['rewards']) for d in data])
        mean_return, std_return = np.mean(returns), np.std(returns)
        max_return, min_return = np.max(returns), np.min(returns)


        print('Num traj: ', len(data))
        print('Max action: ', np.amax(actions))
        print('Min action: ', np.amin(actions))

        print(f'Mean return: {mean_return:.2f}, std: {std_return:.2f}')
        print(f'Max return: {max_return:.2f}, min: {min_return:.2f}')

        print('-'*50)

if __name__=='__main__':
    main()