import subprocess
import os

experiments = [e for e in os.listdir() if e.startswith('mc')]

for experiment in experiments:
    print(experiment)
    command = f'python {experiment}'
    process = subprocess.Popen(command, shell=True)
    process.wait()