import subprocess
import os
import random

while True:
    while True:
        trace_decay = random.choice([0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
        if not os.path.isfile(f'results/gae_{trace_decay}.txt'):
            break
    command = f'python test_gae.py --trace_decay {trace_decay}'
    process = subprocess.Popen(command, shell=True)
    process.wait()