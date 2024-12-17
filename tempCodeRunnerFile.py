avg_reward = 0.
episodes = 10
eval_reward = 0.
eval_step=0
for _  in range(episodes):
    state = env_e.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env_e.step(action)
        eval_reward += reward
        state = next_state
        eval_step+=1
    avg_reward += eval_reward
avg_reward /= episodes
eval_per_step=eval_step/episodes
print("Test total_numsteps: {},eval_steps {}, Avg. Reward: {}".format(total_numsteps,eval_per_step ,round(avg_reward, 2)))
