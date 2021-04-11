
import time
import subprocess as sp
from typing import Callable
SEED = 2021
FIXED_CMDS = ['python3', 'experiment.py', '--init_sampling_seed', f"{SEED}"]


def send_experiment_cmds(combination_generator: Callable[[list], list],
                         fixed_squeezenet_cmds: list,
                         fixed_resnet_cmds: list) -> None:
    # Resnet commands list
    resnet_cmds = combination_generator(fixed_resnet_cmds)

    # SqueezeNet commands list
    squeezenet_cmds = combination_generator(fixed_squeezenet_cmds)

    # List of all cmds to run
    cmds = squeezenet_cmds + resnet_cmds

    # We run experiments
    start = time.time()
    for cmd in cmds:
        cmd = FIXED_CMDS + cmd
        p = sp.Popen(cmd)
        p.wait()

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))