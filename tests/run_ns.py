import subprocess
import os
import random

experiments = [e for e in os.listdir() if e.startswith('test_ns')]

for experiment in experiments:
    print(experiment)
    command = f'python {experiment}'
    process = subprocess.Popen(command, shell=True)
    process.wait()