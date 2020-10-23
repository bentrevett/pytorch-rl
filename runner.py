import random
import os
import subprocess

while True:
    args = {'env': 'CartPole-v1',
            'seed': 1234,
            'n_layers': random.choice([1,2]),
            'grad_clip': random.choice([5, 1, 0.5, 0.1]),
            'hid_dim': random.choice([32, 64, 128, 256]),
            'init': random.choice(['xavier', 'kaiming']),
            'n_runs': 5,
            'n_episodes': random.choice([500, 1000, 2500, 5000, 10_000]),
            'discount_factor': random.choice([0.9, 0.99, 0.999]),
            'start_epsilon': 1.0,
            'end_epsilon': random.choice([0.01, 0.05]),
            'exploration_time': random.choice([0.8, 0.6, 0.4, 0.2]),
            'optim': random.choice(['adam', 'rmsprop']),
            'lr': random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
            }

    name = '-'.join([f'{k}={v}' for k, v in args.items()])

    if os.path.exists('checkpoints/'+name+'_train.pt'):
        continue

    cmd_args = ['python', 'q_learning.py']
    for k, v in args.items():
        cmd_args.append(f'--{k}')
        cmd_args.append(f'{v}')

    print(cmd_args)

    subprocess.run(cmd_args)