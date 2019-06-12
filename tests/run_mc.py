import subprocess
import os
import random

experiments = [e for e in os.listdir() if e.startswith('test_mc')]

for experiment in experiments:
    print(experiment)
    command = f'python {experiment} --dropout 0.5'
    process = subprocess.Popen(command, shell=True)
    process.wait()