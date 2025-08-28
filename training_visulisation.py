import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

class TrainingVisualizer:
    def __init__(self, window_size=100):
        """
        Initialize the training visualizer
        
        Args:
            window_size: Size of the moving average window for smoothing rewards
        """
        self.episodes = []
        self.epsilons = []
        self.rewards = []
        self.best_rewards = []
        self.moving_avg_rewards = []
        self.reward_window = deque(maxlen=window_size)
        self.best_reward_so_far = float('-inf')
        
        # Set up the plot
        plt.ion()  
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('DQN Flappy Bird Training Progress', fontsize=16, fontweight='bold')
        
        # Initialize 
        self.epsilon_line, = self.ax1.plot([], [], 'b-', label='Epsilon', linewidth=2)
        self.reward_line, = self.ax2.plot([], [], 'g-', alpha=0.3, label='Episode Reward', linewidth=1)
        self.moving_avg_line, = self.ax2.plot([], [], 'r-', label=f'Moving Avg ({window_size} eps)', linewidth=2)
        self.best_reward_line, = self.ax2.plot([], [], 'gold', marker='*', markersize=8, 
                                               linestyle='-', linewidth=2, label='Best Reward So Far')
        
        
        self._setup_axes()
        
        
        self.ax1.legend(loc='upper right')
        self.ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _setup_axes(self):
        
        # Epsilon plot (top)
        self.ax1.set_title('Exploration Rate (Epsilon) Over Time', fontweight='bold')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Epsilon', color='blue')
        self.ax1.tick_params(axis='y', labelcolor='blue')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(0.00, 0.105)
        
        # Reward plot (bottom)
        self.ax2.set_title('Training Rewards Over Time', fontweight='bold')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Reward')
        self.ax2.grid(True, alpha=0.3)
    
    def update(self, episode, epsilon, reward):
        
       # Update the visualisatin with new training data
        
        
        self.episodes.append(episode)
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        
        # Update best reward tracking
        if reward > self.best_reward_so_far:
            self.best_reward_so_far = reward
        self.best_rewards.append(self.best_reward_so_far)
        
        # Calculate moving average
        self.reward_window.append(reward)
        moving_avg = np.mean(list(self.reward_window))
        self.moving_avg_rewards.append(moving_avg)
        
        # Update plots every 10 episodes for better performance
        if episode % 10 == 0 or episode <= 50:
            self._update_plots()
    
    def _update_plots(self):
      
        # Update epsilon plot
        self.epsilon_line.set_data(self.episodes, self.epsilons)
        
        # Update reward plots
        self.reward_line.set_data(self.episodes, self.rewards)
        self.moving_avg_line.set_data(self.episodes, self.moving_avg_rewards)
        self.best_reward_line.set_data(self.episodes, self.best_rewards)
        
        # Auto-scale axes
        if len(self.episodes) > 0:
            # Epsilon plot
            self.ax1.set_xlim(0, max(self.episodes) * 1.05)
            
            # Reward plot
            self.ax2.set_xlim(0, max(self.episodes) * 1.05)
            
            # Set y-limits for reward plot with  padding
            min_reward = min(min(self.rewards), min(self.best_rewards))
            max_reward = max(max(self.rewards), max(self.best_rewards))
            padding = abs(max_reward - min_reward) * 0.1 + 1
            self.ax2.set_ylim(min_reward - padding, max_reward + padding)
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_plot(self, filename='training_progress.png', dpi=300):

        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    def save_data(self, filename='training_data.npz'):

        np.savez(filename,
                episodes=np.array(self.episodes),
                epsilons=np.array(self.epsilons), 
                rewards=np.array(self.rewards),
                best_rewards=np.array(self.best_rewards),
                moving_avg_rewards=np.array(self.moving_avg_rewards))
        print(f"Training data saved as {filename}")
    
    def load_data(self, filename='training_data.npz'):
 
        try:
            data = np.load(filename)
            self.episodes = data['episodes'].tolist()
            self.epsilons = data['epsilons'].tolist()
            self.rewards = data['rewards'].tolist() 
            self.best_rewards = data['best_rewards'].tolist()
            self.moving_avg_rewards = data['moving_avg_rewards'].tolist()
            
            # Update best reward tracking
            if len(self.best_rewards) > 0:
                self.best_reward_so_far = self.best_rewards[-1]
            
            # Update reward window
            window_data = self.rewards[-len(self.reward_window):] if len(self.rewards) > 0 else []
            self.reward_window.clear()
            self.reward_window.extend(window_data)
            
            self._update_plots()
            print(f"Training data loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with new data.")
    
    def close(self):
        
        plt.ioff()
        plt.close(self.fig)


