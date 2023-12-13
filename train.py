from typing import Tuple, Dict

import gym
import torch
import numpy as np

import pathlib
import time

"""
### Lunar Lander Environment ###
* Action space:
    - The agent has four actions available:
        * Do nothing = 0
        * Fire right engine = 1
        * Fire main engine = 2
        * Fire left engine = 3
        
* Observation space:
    - Agent's observation space is a state vector with 8 variables:
        * (x, y) coordinates. Landing pad is always (0, 0)
        * Linear velocity. xdot, ydot
        * Angle. Theta
        * Angular velocity. Thetadot
        * l, r. Left and Right legs are grounded or not.
        
* Rewards:
    - At every episode reward is granted. Total reward for corresponding episode is
    sum of step rewards for that episode.
    - Rewards:
        * Distance to landing pad
        * Lander speed
        * Lander is tilted or not
        * 10 points if each leg is on the ground
        * Decreased 0.03 points for each side engine fired
        * Decreased 0.3 points for main engine fired 
        * Additional -100 / +100 for crushed / landed safely
        
* Terminal State:
    - Episodes ends at a terminal state.
    - The state is considered terminal if following:
        * Lunar lander crashes
        * Abs(x) > 1. (Goes beyond the borders)
"""

class QValue(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.dense_layer_1 = torch.nn.Linear(state_size, 128)
        self.dense_layer_2 = torch.nn.Linear(128, action_size)
    
    def forward(self, state):
        features = torch.relu(self.dense_layer_1(state))
        return self.dense_layer_2(features)

class FixedSizeBuffer:
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.items = []
        self.buffer_size = 0
    
    def append(self, item):
        if self.buffer_size >= self.max_size:
            self.items[self.buffer_size % self.max_size] = item
        else:
            self.items.append(item)
        self.buffer_size += 1
    
    def __len__(self):
        return len(self.items)
        
# Model saving directory
save_dir = pathlib.Path().resolve().joinpath('models')

# Seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
memory_size = 10000         # Buffer memory size
alpha = 1e-3                # Learning rate
gamma = 0.9                # Dicount factor
n_iteration = 5000000        # Total number of iteration
epsilon = 0.15               # Randomness
batch_size = 64             # Amount of SARSA for learning
iter_checkpoint = 250 * 1000 # Save point

# Experience Buffer
buffer = FixedSizeBuffer(max_size=memory_size)

# Load the environment
env = gym.make('LunarLander-v2')
env.reset(seed=seed)

# Neural Network
value_function = QValue(state_size=8, action_size=4)
optimizer = torch.optim.Adam(value_function.parameters(), lr=alpha)

def sample_from_buffer(batch_size: int) -> Dict[str, torch.Tensor]:
    """Take sample from buffer

    Args:
        batch_size (int): how much SARSA?

    Returns:
        Dict[str, torch.Tensor]: SARSA's
    """
    indexes = np.random.randint(0, len(buffer), batch_size)
    items = [buffer.items[index] for index in indexes]
    experince_dict = {
        key: torch.from_numpy(np.array([item[key] for item in items])).float()
        for key in items[0].keys()
    }
    return experince_dict

def get_qvalues(state: np.ndarray) -> torch.Tensor:
    """Get qvalues from neural network

    Args:
        state (np.ndarray): current state

    Returns:
        torch.Tensor: state after neural network run
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    qvalues = value_function.forward(state)
    return qvalues

def get_action(state: np.array, epsilon: float):
    """Get action from state and epsion

    Args:
        state (np.array): current state
        epsilon (float): randomness ratio
    """
    if np.random.rand() > epsilon:
        return np.random.randint(0, 4)
    qvalues = get_qvalues(state=state)
    _, max_index = qvalues.max(dim=1)
    action = max_index.item()
    return action

def td_loss_fn(sample: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Loss calculation

    Args:
        sample (Dict[str, torch.Tensor]): SARSA
    
    Returns:
        torch.Tensor: TD error tensor
    """
    action = sample['action'].long().reshape(-1, 1)
    reward = sample['reward'].long().reshape(-1, 1)
    next_action = sample['next_action'].long().reshape(-1, 1)
    done = sample['done'].long().reshape(-1, 1)
    
    next_qvalues = value_function.forward(sample['next_state'])
    selected_next_qvalues = next_qvalues.gather(dim=1, index=next_action)

    target_qvalues = reward + (1 - done) * gamma * selected_next_qvalues
    qvalues = value_function.forward(sample["state"]).gather(dim=1, index=action)
    
    td_loss = ((target_qvalues.detach() - qvalues) ** 2) 
    return td_loss.mean()

def evaluate() -> float:
    """Evaluate the soft agent for one episode

    Args:
        is_render (bool, optional): render or not. Defaults to False.
        is_save_render (bool, optional): save the render or not. Defaults to False.

    Returns:
        float: episode reward
    """
    eval_env = gym.make('LunarLander-v2')
    
    eps_reward = 0
    state, _ = eval_env.reset()
    done = False
    while not done:
        action = get_action(state, epsilon)
        state, reward, done, _, _ = eval_env.step(action)
        eps_reward += reward
    return eps_reward

if __name__ == '__main__':
    episodic_rewards = []
    episode_reward = 0
    state, _ = env.reset()
    action = get_action(state, epsilon)
    # time_holder= time.time()
    for iter_index in range(1, n_iteration + 1):
        next_state, reward, done, _, _ = env.step(action=action)
        next_action = get_action(state=next_state, epsilon=epsilon)
        episode_reward += reward
        buffer.append(
            {
                'state':state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'next_action': next_action,
                'done': done
            }
        )
        action, state = next_action, next_state
        # if (time.time() - time_holder > 30): done = 1 
        
        if done:
            # Episode finished
            episodic_rewards.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()
            # time_holder = time.time()
        
        if not(iter_index % iter_checkpoint):
            # Save the model for every iter_checkpoint iteration
            torch.save({
                'iteration': iter_index,
                'model_state_dict': value_function.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': td_loss.item()
            }, save_dir.joinpath(f'model{iter_index}.pt'))
        
        if not(iter_index % 50000):
            # Update user about training
            eval_reward = evaluate()
            avg_train_reward = np.mean(episodic_rewards)
            episodic_rewards = []
            print(f'Iteration: {iter_index}, Evaluation: {eval_reward:.4f}, Training: {avg_train_reward:.4f} Rewards')
        
        if len(buffer) > 1000:
            # Learning
            sample = sample_from_buffer(batch_size=batch_size)
            optimizer.zero_grad()
            td_loss = td_loss_fn(sample)
            td_loss.backward()
            optimizer.step()
        