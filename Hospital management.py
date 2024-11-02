import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

class HospitalEnv:
    def __init__(self, num_doctors):
        self.num_doctors = num_doctors  # Number of available doctors
        self.reset()

    def reset(self):
        self.time = 0  # Current time
        self.waiting_patients = []  # Queue of waiting patients
        self.current_patients = 0  # Currently treated patients
        self.total_wait_time = 0  # Total wait time for all patients
        self.num_patients_served = 0  # Number of patients served
        return self.get_state()

    def get_state(self):
        # State consists of current time, number of waiting patients, and currently treated patients
        return np.array([self.time, len(self.waiting_patients), self.current_patients])

    def step(self, action):
        # Action: number of doctors to allocate (0 to num_doctors)
        action = np.clip(action, 0, self.num_doctors)  # Ensure action is within bounds
        treated_patients = min(action, len(self.waiting_patients))  # Treat as many patients as available doctors
        
        # Update the waiting patients list
        self.total_wait_time += len(self.waiting_patients)  # Increase total wait time by number of waiting patients
        self.current_patients += treated_patients  # Update currently treated patients
        self.num_patients_served += treated_patients  # Update patients served
        
        # Remove treated patients from the waiting list
        self.waiting_patients = self.waiting_patients[treated_patients:]

        # Increase time and generate new patients (1 new patient per time unit)
        self.time += 1
        self.waiting_patients.append(1)  # New patient arrives
        
        # Calculate reward: Negative of average wait time
        if self.num_patients_served > 0:
            avg_wait_time = self.total_wait_time / self.num_patients_served
        else:
            avg_wait_time = 0
        reward = -avg_wait_time
        
        # Check if done: Arbitrary limit for simulation
        done = self.time >= 50  # Simulate for 50 time units

        next_state = self.get_state()  # Get the next state
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
            return self.actions[np.argmax(self.q_table.get(state_key, [0] * len(self.actions)))]

    def learn(self, state, action, reward, next_state):
        state_key = str(state)
        next_state_key = str(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * len(self.actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0] * len(self.actions)
        
        # Q-learning update rule
        q_value = self.q_table[state_key][action]
        max_future_q = max(self.q_table[next_state_key])
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - q_value)
        self.q_table[state_key][action] = new_q_value

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

def run_simulation(num_doctors):
    env = HospitalEnv(num_doctors)
    agent = QLearningAgent(actions=list(range(num_doctors + 1)))  # Actions: 0 to num_doctors

    episodes = 1000  # Number of training episodes
    avg_wait_times = []  # To store average wait times for visualization

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_wait_time = 0
        total_patients_served = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            
            # Accumulate wait time and patients served for average calculation
            total_wait_time += -reward  # Since reward is negative average wait time
            total_patients_served += 1

        if total_patients_served > 0:
            avg_wait_times.append(total_wait_time / total_patients_served)
        else:
            avg_wait_times.append(0)

    return agent, avg_wait_times

# Run the simulation with a specific number of doctors
num_doctors = 5
trained_agent, avg_wait_times = run_simulation(num_doctors)

# Visualization of Average Wait Time Over Episodes
plt.figure(figsize=(10, 5))
plt.plot(avg_wait_times, label='Average Wait Time', color='blue')
plt.title('Average Wait Time per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Wait Time')
plt.grid(True)
plt.legend()
plt.show()
