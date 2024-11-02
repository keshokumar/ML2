import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

class PortfolioEnv:
    def __init__(self, data):
        self.data = data
        self.num_steps = len(data)
        self.current_step = 0
        self.total_portfolio_value = 1000  # Initial portfolio value
        self.assets_allocation = [0.5, 0.5]  # Initial allocation: 50% Asset A, 50% Asset B

    def reset(self):
        self.current_step = 0
        self.total_portfolio_value = 1000
        self.assets_allocation = [0.5, 0.5]
        return self._get_state()

    def _get_state(self):
        # Ensure the current step is within bounds
        if self.current_step < self.num_steps:
            return np.array(self.data.iloc[self.current_step].values)
        else:
            return np.array(self.data.iloc[-1].values)  # Return the last state's values

    def step(self, action):
        # Action: 0 = invest more in Asset A, 1 = invest more in Asset B
        if action == 0:
            self.assets_allocation[0] += 0.1
            self.assets_allocation[1] -= 0.1
        elif action == 1:
            self.assets_allocation[0] -= 0.1
            self.assets_allocation[1] += 0.1
        
        # Ensure allocations are within bounds
        self.assets_allocation = np.clip(self.assets_allocation, 0, 1)
        self.assets_allocation /= sum(self.assets_allocation)  # Normalize
        
        # Calculate new portfolio value
        price_A = self.data.iloc[self.current_step]['Asset_A']
        price_B = self.data.iloc[self.current_step]['Asset_B']
        self.total_portfolio_value = (self.total_portfolio_value * self.assets_allocation[0] * price_A + 
                                       self.total_portfolio_value * self.assets_allocation[1] * price_B)
        
        # Move to the next step
        self.current_step += 1
        
        # Check if the episode has finished
        done = self.current_step >= self.num_steps
        
        if done:
            # Reward is the final portfolio value when done
            reward = self.total_portfolio_value
        else:
            # Reward can be set to zero for interim steps
            reward = 0
        
        # Return next state, reward, and done flag
        next_state = self._get_state()  # Get the next state
        return next_state, reward, done

class QLearningAgent:
    def __init__(self, actions):
        self.q_table = {}
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.1

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)  # Explore
        else:
            state_key = str(state)
            return self.actions[np.argmax(self.q_table.get(state_key, [0, 0]))]  # Exploit

    def learn(self, state, action, reward, next_state):
        state_key = str(state)
        next_state_key = str(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0, 0]
        
        # Q-learning update rule
        q_value = self.q_table[state_key][action]
        max_future_q = max(self.q_table[next_state_key])
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - q_value)
        self.q_table[state_key][action] = new_q_value

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

def run_simulation(data):
    env = PortfolioEnv(data)
    agent = QLearningAgent(actions=[0, 1])  # Actions: invest more in Asset A (0) or Asset B (1)

    episodes = 1000  # Number of training episodes
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

    return agent

# Create a sample dataset
data = pd.DataFrame({
    'Asset_A': np.random.uniform(100, 200, 100),  # Simulated prices for Asset A
    'Asset_B': np.random.uniform(100, 200, 100)   # Simulated prices for Asset B
})

# Run the simulation to train the agent
trained_agent = run_simulation(data)

# Real-time demonstration of the agent's investment decisions
portfolio_value = 1000
portfolio_values = []  # List to store portfolio values for visualization
actions_taken = []  # List to store actions taken

for i in range(len(data)):
    state = data.iloc[i].values
    action = trained_agent.choose_action(state)
    
    # Update portfolio based on action
    if action == 0:  # Invest more in Asset A
        portfolio_value += portfolio_value * 0.05  # Assume 5% return
    else:  # Invest more in Asset B
        portfolio_value += portfolio_value * 0.03  # Assume 3% return

    portfolio_values.append(portfolio_value)  # Record portfolio value
    actions_taken.append(action)  # Record action taken
    print(f"Step {i}, Action: {'Invest in Asset A' if action == 0 else 'Invest in Asset B'}, "
          f"Portfolio Value: {portfolio_value:.2f}")
    time.sleep(0.5)  # Pause for half a second for better visualization

# Visualization of Portfolio Value Over Time
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label='Portfolio Value', color='blue')
plt.title('Portfolio Value Over Time')
plt.xlabel('Step')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.legend()
plt.show()
