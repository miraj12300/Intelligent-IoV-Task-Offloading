def evaluate_model(agent, env, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = agent.get_action(state)
            state, reward, done = env.step(action[0], action[1])
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)
