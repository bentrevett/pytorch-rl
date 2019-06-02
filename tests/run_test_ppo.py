import os
import subprocess
import random

while True:
    while True:
        ppo_steps = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ppo_clip = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        if not os.path.isfile(f'results/ppo_steps={ppo_steps}_clip={ppo_clip}.txt'):
            break
    
    process = subprocess.Popen(f'python test_ppo.py --ppo_steps {ppo_steps} --ppo_clip {ppo_clip}', shell=True, stdout=subprocess.PIPE)
    process.wait()
