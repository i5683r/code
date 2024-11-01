import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Env_normalize import env
import itertools
import pandas as pd
import openpyxl
import scipy.io as io
import random
wb = openpyxl.Workbook()
tf.compat.v1.reset_default_graph()
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
ring = wb.active

from agent_ddpg import DDPG
sess = tf.Session()

def get_agents_action(observation_n, sess, noise_rate):
    actions = []
    for i, obs in enumerate(observation_n):
        agent_name = f'agent{i+1}'
        action = agents[agent_name].action(np.array([obs]), sess) + np.random.randn(1) * noise_rate
        actions.append(action)
    return actions

def train_agent(agent, memory, sess, n, policy_delay):
# def train_agent(agent, memory, sess, n):
    batch = memory.sample(n)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    action_batch = action_batch.reshape(n, 3)
    reward_batch = reward_batch.reshape(n, 1)
    done_batch = done_batch.reshape(n, 1)

    next_actions = agent.target_action(next_state_batch, sess)
    noise = np.clip(np.random.normal(0, 0.2, size=np.shape(next_actions)), -0.5, 0.5)
    next_actions = np.clip(next_actions + noise, -1, 1)

    target_q1 = agent.Q(next_state_batch, next_actions, sess, target=True)
    target_q = reward_batch + 0.9995 * target_q1 * (1 - done_batch)

    target_q = target_q.reshape(-1, 1)  # Ensure target_q has shape (n, 1)
    q_values = agent.train_critic(state_batch, action_batch, target_q, sess)

    agent.train_actor(state_batch, sess)
    agent.update_target_network(sess)
    return q_values

num_agents = 4
rounds = 3000
T = 24
batch_size = 48
noise_rate = 0.02
ep_reward = np.zeros((rounds, T))
all_reward = np.zeros(rounds)
train_point = 0
v_c = np.zeros((num_agents, T))
if __name__ == '__main__':
    with tf.compat.v1.Session() as sess:

        agents = {}
        agents_target = {}
        policy_delay = 0

        for i in range(1, 6):
            agent_name = f'agent{i}'
            agents[agent_name] = DDPG(agent_name)

        agent_names = list(agents.keys())
        Num_agents = len(agent_names)

        sess.run(tf.compat.v1.global_variables_initializer())

        for agent in agents.values():
            agent.update_target_network(sess)

        for round in range(rounds):
            total_reward = 0
            current_states = env.reset()
            if round % 200 == 0:
                noise_rate *= 0.95
            for t in range(T):
                current_states = env.get_states()
                actions = get_agents_action(current_states, sess, noise_rate)

                next_states, rewards, done = env.step(actions)
                ep_reward[round, t] = sum(rewards[:])

                for i, agent_name in itertools.islice(enumerate(agents), 4):
                    agents[agent_name].memory.add(current_states[i], actions[i], rewards[i],
                                                               next_states[i], done)
                    if round >= train_point:
                        train_agent(agents[agent_name], agents[agent_name].memory, sess, batch_size, policy_delay)
                policy_delay += 1

            for agent in agents.values():
                agent.update_target_network(sess)

            if round % 50 == 0:
                agents[agent_names[4]].backup_parameters(agents[agent_names[1]], sess)
                agents[agent_names[1]].copy_parameters(agents[agent_names[3]], sess)
                agents[agent_names[3]].copy_parameters(agents[agent_names[0]], sess)
                agents[agent_names[0]].copy_parameters(agents[agent_names[2]], sess)
                agents[agent_names[2]].copy_parameters(agents[agent_names[4]], sess)

            all_reward[round] = sum(ep_reward[round,:])
            if round % 500 == 0:
            print(f'Round {round}: Total Reward: {all_reward[round]}')


