import os
import subprocess
import random

while True:
    while True:
        n_envs = random.choice([1, 2, 4, 8, 16])
        hidden_dim = random.choice([32, 64, 128, 256])
        n_layers = random.choice([0, 1, 2])
        activation = random.choice(['relu', 'tanh', 'sigmoid'])
        dropout = random.choice([0.0, 0.1, 0.25, 0.5])
        lr = random.choice([0.05, 0.01, 0.005])
        discount_factor = random.choice([0.9, 0.99, 0.999])
        grad_clip = random.choice([0.1, 0.25, 0.5, 1.0])
        n_steps = random.choice([5, 10, 25, 50, 100, 200, 500])
        params = f'ns_a2c_{n_envs}ne_{hidden_dim}hd_{n_layers}nl_{activation}ac_{dropout}do_{lr}lr_{discount_factor}df_{grad_clip}gc_{n_steps}ns'
        if not os.path.exists(f'results/{params}.txt'):
            break
        command = f'python ns_a2c --n_envs {n_envs} --hidden_dim {hidden_dim} --n_layers {n_layers} --activation {activation} --dropout {dropout} --lr {lr} --discount_factor {discount_factor} --grad_clip {grad_clip} --n_steps {n_steps}'
        process = subprocess.Popen(command, shell=True)
        process.wait()
    

