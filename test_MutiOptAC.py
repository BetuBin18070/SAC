import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Walker2d-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
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
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
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


def main(buff_len):
    args = parser.parse_args()
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    #Tesnorboard
    writer = SummaryWriter('runs/test_offline/{}_SAC_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else "","updates_per_step"+str(int(args.updates_per_step)),"buff"+str(int(buff_len)))) 

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory.load_buffer("checkpoints/random/sac_buffer_Walker2d-v3_"+str(int(buff_len)))
    updates = 0
    eval_per_step = 1000
    for i in range(200000):
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, critic_grad_norm, action_grad_norm, next_q_value = agent.update_parameters(memory, args.batch_size, updates)
        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
        writer.add_scalar('loss/policy', policy_loss, updates)
        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        writer.add_scalar('grad_norm/critic', critic_grad_norm, updates)
        writer.add_scalar('grad_norm/action', action_grad_norm, updates)
        writer.add_scalar('next_q_value', next_q_value, updates)
        updates += 1

        if i % eval_per_step == 0:
            avg_reward = 0.
            episodes = 10
            eval_reward = 0.
            for _  in range(episodes):
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = env.step(action)
                    eval_reward += reward
                    state = next_state
                avg_reward += eval_reward
            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, i)

            print("----------------------------------------")
            #print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("Test total_numsteps: {}, Avg. Reward: {}".format(i, round(avg_reward, 2)))
            print("----------------------------------------")
            if avg_reward > 3000:
                agent.save_checkpoint(args.env_name, str(i))
                exit(0)


if __name__ == "__main__":

    main(10000)

