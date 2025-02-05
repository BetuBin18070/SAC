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
from wrapers import wrap_gym

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
log_dir = 'runs/{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "",'seed'+str(args.seed))
writer = SummaryWriter(log_dir)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
eval_per_step = 20000




for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size and total_numsteps>=args.start_steps:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
     
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


        next_state, reward, done, _ = env.step(action) # Step
        # if total_numsteps>=args.start_steps:
        #     env.render()
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        #mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = 1 if episode_steps == 1000 else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

        # if total_numsteps == 40000 or total_numsteps == 100000 or total_numsteps == 200000 or total_numsteps ==300000 or total_numsteps ==400000 :
        #     agent.save_checkpoint(args.env_name, suffix=str(total_numsteps))
        #     memory.save_buffer(args.env_name, suffix=str(total_numsteps))
        #     if total_numsteps ==400000:
        #         exit(0)

        if total_numsteps % 5000 == 0:
            #print("total_numsteps",total_numsteps)
            agent.save_checkpoint(args.env_name+str(int(args.seed)), suffix=str(total_numsteps))


            


        if total_numsteps % eval_per_step == 0 and args.eval is True and total_numsteps >= args.start_steps:
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


            writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)

            print("----------------------------------------")
            #print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("Test total_numsteps: {}, Avg. Step: {} Reward: {}".format(total_numsteps, round(avg_steps,2),round(avg_reward, 2)))
            print("----------------------------------------")
            # if total_numsteps % 10000 == 0:
            #     agent.save_checkpoint(args.env_name, str(int(total_numsteps)))




    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward,total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    #if i_episode % 10 == 0 and args.eval is True:
    #print('t/eva',total_numsteps % eval_per_step)

env.close()

