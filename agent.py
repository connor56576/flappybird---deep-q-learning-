import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Hyperparameters - tweak during training
GAMMA = 0.99 # long reward/ short reward discount value
LR = 1e-5 # learning rate
BATCH_SIZE = 64 #number of replays in smaple
REPLAY_CAPACITY = 100_000 
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 500        # update target network
EPS_START = 0.1
EPS_END = 0.01      
EPS_DECAY_STEPS = 200000       # how fast epsilon goes donw
MAX_EPISODES = 5000
MAX_STEPS_PER_EP = 5000
MODEL_PATH = "dqn_flappy.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#uses numpy circular arrays (fifo)
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        states = torch.tensor(self.states[idx], dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(self.actions[idx], dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(self.rewards[idx], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(self.next_states[idx], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(self.dones[idx], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


# Simple Q-network 1 hidden layer 128 nodes 
class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        lr=LR,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        replay_capacity=REPLAY_CAPACITY,
        min_replay_size=MIN_REPLAY_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        double_dqn=True
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = DEVICE

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)  #policy network
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)  #target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) #uses adam as optimsizer
        self.replay = ReplayBuffer(replay_capacity, state_dim)
        self.min_replay_size = min_replay_size
        self.double_dqn = double_dqn
        self.update_count = 0
        self.target_update_freq = target_update_freq

        # epsilon decay (linear)
        self.eps = EPS_START
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay_steps = EPS_DECAY_STEPS
        self.total_steps = 0

    def select_action(self, state, eval_mode=False):
        if eval_mode:
            eps_use = 0.0
        else:
            eps_use = self.eps
        if random.random() < eps_use:
            return random.randrange(self.n_actions)
        else:
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.policy_net(s)
                return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update_epsilon(self):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / max(1, self.eps_decay_steps))
        self.eps = self.eps_start + frac * (self.eps_end - self.eps_start)

    def learn(self):
        if len(self.replay) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # current Q
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            q_target = rewards + (1.0 - dones) * (self.gamma * next_q)   #bellman equation 

        loss = nn.functional.mse_loss(q_values, q_target) # mse loss function, good for flapyp bird

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.detach().cpu().item())

    def save(self, path=MODEL_PATH):    
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path=MODEL_PATH):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state'])
        self.target_net.load_state_dict(checkpoint['target_state'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])


#  Environment wrapper, normalizes values from flappy game 
class EnvWrapper:     
    def __init__(self, flappy_env):
        self.env = flappy_env

    def reset(self):
        s = self.env.reset()
        return self._normalize_state(s)

    def step(self, action):
        next_s, reward, done = self.env.step(action, False)
        return self._normalize_state(next_s), reward, done, {}

    def render(self):
        try:
            self.env.render()
        except Exception:
            pass

    def _normalize_state(self, s):
        player_y = s[0] / float(720)
        player_vel = (s[1] + 1000) / 2000.0
        dist = s[2] / float(1280)
        top_norm = s[3] / float(720)
        bottom_norm = s[4] / float(720)
        return np.array([player_y, player_vel, dist, bottom_norm, top_norm], dtype=np.float32)


# Training loop 
def train(agent: DQNAgent, env_wrapper: EnvWrapper, num_episodes=MAX_EPISODES, render=False, save_every=500, visulise=True):
    if visulise:
        visualiser = TrainingVisualizer()
    best_reward = -1000
    stats = {'episode': [], 'reward': [], 'loss': []}
    total_steps = 0
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        state = env_wrapper.reset()
        ep_reward = 0.0
        losses = []
        for t in range(MAX_STEPS_PER_EP):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, _ = env_wrapper.step(action)

            agent.store(state, action, reward, next_state, float(done))
            agent.update_epsilon()

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_reward += reward
            total_steps += 1

            if render:
                env_wrapper.render()

            if done:
                break

        avg_loss = float(np.mean(losses)) if len(losses) > 0 else None
        elapsed = time.time() - start_time
        if visulise:
            visualiser.update(ep, agent.eps, ep_reward)

        print(f"[Ep {ep}] reward={ep_reward:.2f}, steps={t+1}, eps={agent.eps:.4f}, avg_loss={avg_loss}, total_steps={total_steps}, elapsed={int(elapsed)}s")
        stats['episode'].append(ep)
        stats['reward'].append(ep_reward)
        stats['loss'].append(avg_loss)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(f"dqn_flappy_best.pth")
            print(f"New best reward: {best_reward:.2f}! Model saved.")

        if ep % save_every == 0:
            agent.save(f"dqn_flappy_ep{ep}.pth")
            print(f"Saved model at episode {ep}")

    agent.save(MODEL_PATH)
    print("Training complete. Model saved to", MODEL_PATH)
    return stats


if __name__ == "__main__":
    from Flappy import FlappyEnv
    from training_visulisation import TrainingVisualizer
    env = FlappyEnv()
    wrapper = EnvWrapper(env)
    state_dim = 5
    n_actions = 2

    agent = DQNAgent(state_dim, n_actions)
    if os.path.exists(MODEL_PATH):
        print("Loading model:", MODEL_PATH)
        agent.load(MODEL_PATH)

   # change these values for different modes (render and visualise)
   # change num_episodes for shorter training runs
    stats = train(agent, wrapper, num_episodes=5000, render=False, save_every=500, visulise=True) 
