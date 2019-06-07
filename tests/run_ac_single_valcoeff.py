import subprocess
import os
import random

while True:
    while True:
        hidden_dim = random.choice([8, 16, 32, 64, 128, 256])
        if not os.path.isfile(f'results/ac_single_valcoeff_{hidden_dim}.txt'):
            break
    command = f'python test_ac_single_valcoeff.py --hidden_dim {hidden_dim}'
    process = subprocess.Popen(command, shell=True)
    process.wait()