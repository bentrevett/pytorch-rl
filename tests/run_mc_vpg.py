import subprocess
import os
import random

while True:
    while True:
        hidden_dim = random.choice([64, 128, 256])
        dropout = random.choice([0, 0.25, 0.5])
        lr = random.choice([0.05, 0.01, 0.005])
        init = random.choice(['none', 'xavier-uniform', 'xavier-normal'])
        activation = random.choice(['relu', 'tanh'])
        if not os.path.exists(f'results/mc_vpg_{hidden_dim}_{dropout}_{lr}_{init}_{activation}.txt'):
            break
    command = f'python search_mc_vpg.py --hidden_dim {hidden_dim} --dropout {dropout} --lr {lr} --init {init} --activation {activation}'
    print(command)
    process = subprocess.Popen(command, shell=True)
    process.wait()