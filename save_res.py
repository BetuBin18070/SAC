import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import dm2gym
import os
from wrapers import wrap_gym


def log_results(step, step_reward, filepath='res.txt'):
    with open(filepath, 'a') as f:
        f.write(f"{step},{step_reward}\n")


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Hopper-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2 Swimmer-v3)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

env_e = gym.make(args.env_name)
env_e.seed(args.seed)
env_e.action_space.seed(args.seed)

print(env.action_space)
print(env.observation_space)
if 'v0' in args.env_name:
    env = wrap_gym(env)
    env_e = wrap_gym(env_e)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
log_dir = f'./checkpoints/{args.env_name}{args.seed}'
    
res_file_path = os.path.join('./checkpoints', f'sac_res_{args.env_name}_{args.seed}.txt')

if not os.path.exists(res_file_path):
        with open(res_file_path, 'w') as f:
            f.write("0,0\n")  # 可选：写入表头


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
eval_per_step = 20000


idx_list = list(range(5000,args.num_steps+1,5000))

for idx in idx_list:
    agent.load_checkpoint(f'./checkpoints/{args.env_name}{args.seed}/sac_checkpoint_{args.env_name}{args.seed}_{idx}',evaluate=True)

    avg_reward = 0.
    avg_steps = 0.
    episodes = 10
    for _  in range(episodes):
        state = env_e.reset()
        done = False
        eval_reward = 0.
        eval_step=0
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env_e.step(action)
            #env_e.render()
            eval_reward += reward
            state = next_state
            eval_step+=1
        
        print("eval_reward",eval_reward)
        avg_reward += eval_reward
        avg_steps+=eval_step
    avg_reward /= episodes
    avg_steps/=episodes
    log_results(idx, avg_reward, res_file_path)
