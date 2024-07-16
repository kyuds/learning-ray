# https://maxpumperla.com/learning_ray/ch_03_core_app

import random
import os
import time
import ray
import numpy as np

###############################################################################

# Environment

class Discrete:
    def __init__(self, num_actions: int):
        # Discrete(4) --> up, down, left, right
        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n - 1)

class Environment:
    def __init__(self, *args, **kwards):
        self.seeker, self.goal = (0, 0), (4, 4)
        self.info = {
            "seeker": self.seeker,
            "goal": self.goal
        }
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5 * 5)
    
    def reset(self):
        self.seeker = (0, 0)
    
    def get_observation(self):
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        return 1 if self.seeker == self.goal else 0
    
    def is_done(self):
        return self.seeker == self.goal

    def step(self, action):
        if action == 0:  # move down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("Invalid action")

        obs = self.get_observation()
        rew = self.get_reward()
        done = self.is_done()
        return obs, rew, done, self.info

    def render(self):
        grid = [['| ' for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        print(''.join([''.join(grid_row) for grid_row in grid]))

###############################################################################

# Policy & Simulation

class Policy:
    def __init__(self, env):
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n)
        ]
        self.action_space = env.action_space
    
    def get_action(self, state, explore=True, epsilon=0.1):
        if explore and random.uniform(0, 1) < epsilon:
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])

class Simulation(object):
    def __init__(self, env):
        self.env = env
    
    def rollout(self, policy, render=False, explore=True, epsilon=0.1):
        experiences = []
        state = self.env.reset()
        done = False
        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            experiences.append([state, action, reward, next_state])
            state = next_state
            if render:
                time.sleep(0.05)
                self.env.render()
        return experiences

def update_policy(policy, experiences, weight=0.1, discount_factor=0.9):
    for state, action, reward, next_state in experiences:
        next_max = np.max(policy.state_action_table[next_state])
        value = policy.state_action_table[state][action]
        new_value = (1 - weight) * value + \
                    weight * (reward + discount_factor * next_max)
        policy.state_action_table[state][action] = new_value

def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    policy = Policy(env)
    sim = Simulation(env)
    for _ in range(num_episodes):
        experiences = sim.rollout(policy)
        update_policy(policy, experiences, weight, discount_factor)

    return policy

def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    policy = Policy(env)
    sim = Simulation(env)
    for _ in range(num_episodes):
        experiences = sim.rollout(policy)
        update_policy(policy, experiences, weight, discount_factor)

    return policy

def evaluate_policy(env, policy, num_episodes=10):
    simulation = Simulation(env)
    steps = 0

    for _ in range(num_episodes):
        experiences = simulation.rollout(policy, render=True, explore=False)
        steps += len(experiences)

    print(f"{steps / num_episodes} steps on average "
          f"for a total of {num_episodes} episodes.")

    return steps / num_episodes

###############################################################################

@ray.remote
class SimulationActor(Simulation):
    def __init__(self):
        super().__init__(Environment())

def train_policy_parallel(env, num_episodes=1000, num_simulations=4):
    policy = Policy(env)
    simulations = [SimulationActor.remote() for _ in range(num_simulations)]

    policy_ref = ray.put(policy)
    for _ in range(num_episodes):
        experiences = [sim.rollout.remote(policy_ref) for sim in simulations]

        while len(experiences) > 0:
            finished, experiences = ray.wait(experiences)
            for xp in ray.get(finished):
                update_policy(policy, xp)
    return policy

environment = Environment()
parallel_policy = train_policy_parallel(environment)
evaluate_policy(environment, parallel_policy)
