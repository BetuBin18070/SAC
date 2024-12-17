import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory


# memory = ReplayMemory(1e6, 5)
# memory.load_buffer("checkpoints/sac_buffer_Walker2d-v3_20000")
# memory.position = 1000
# memory.buffer = memory.buffer[:memory.position]

# memory.save_buffer("Walker2d-v3", str(int(memory.position)))
# print(len(memory.buffer))


memory = ReplayMemory(1e6, 5)
memory.load_buffer("checkpoints/update_ac_10/sac_buffer_Walker2d-v3_17000")

print(len(memory.buffer))
x = memory.buffer[4000:17000]
x = memory.buffer[0:4000]
x = memory.buffer[4000:10000]
x = memory.buffer[10000:17000]
print(type(x)) 
res = 0

for i in range(len(x)):
    res += x[i][2]

print(res/len(x))

#print(np.mean(memory.buffer[4000:17000][2]))
