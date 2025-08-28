import torch
import os
import time
from agent import DQNAgent, EnvWrapper, MODEL_PATH, DEVICE
from Flappy import FlappyEnv   

def evaluate(num_episodes=10, render=True): #testing loop using last saved model from trianing 
    env = FlappyEnv()
    wrapper = EnvWrapper(env)

    state_dim = 5
    n_actions = 2

    agent = DQNAgent(state_dim, n_actions)
    if os.path.exists(MODEL_PATH):
        print("Loading trained model:", MODEL_PATH)
        agent.load(MODEL_PATH)
    else:
        print("No trained model found! Train first with dqn_flappy_agent.py")
        return

    agent.eps = 0.0  # no exploration during evaluation
    rewards = []

    for ep in range(1, num_episodes + 1):
        state = wrapper.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, _ = wrapper.step(action)
            state = next_state
            ep_reward += reward
            steps += 1

            if render:
                wrapper.render()

        print(f"[Eval Ep {ep}] reward={ep_reward:.2f}, steps={steps}")
        rewards.append(ep_reward)

        # small pause between episodes so you can see resets
        time.sleep(1.0)

    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    evaluate(num_episodes=5, render=True)
