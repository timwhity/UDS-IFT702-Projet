from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import csv 
import gym
import os

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    """
    Sélectionne la meilleure action à réaliser. Explore au début (choix de l'action au hasard),
    puis au fur et à mesure que l'agent apprend, se base sur ses connaissances pour choisir la meilleure action.
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # return policy_net(state.to('cuda')).max(1)[1].view(1,1) # avec cuda
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)


def optimize_model():
    """
    Permet de gérer l'apprentissage des poids et calcule la perte Smooth L1
    
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)
    """
    batch = Transition(*zip(*transitions))

    # actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) # avec cuda
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cpu'), batch.action)))
    # rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) # avec cuda
    rewards = tuple((map(lambda r: torch.tensor([r], device='cpu'), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool) # changmeent uint8 to bool

    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                    if s is not None]).to('cuda') # avec cuda
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cpu')


    # state_batch = torch.cat(batch.state).to('cuda') # avec cuda
    state_batch = torch.cat(batch.state).to('cpu')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss

def get_state(obs): 
    """
    Reformate et convertit l'entrée en entrée utilisable par un réseau de neurones pytorch
    """
    state = np.array(obs)
    state = state.transpose((2, 0, 1)) # pour avoir les params Channel, Height, Width dans le bon ordre
    state = torch.from_numpy(state) # convertit le tableau numpy en tenseur pytorch
    return state.unsqueeze(0) # ajoute une nouvelle dimension à la position 0 

def train(env, n_episodes, render=False, max_timesteps=1000):
    steps_done_ = [] 
    episode_ = [] 
    t_ = [] 
    total_reward_ = [] 
    loss_values = [] 
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in range(max_timesteps): # while True : 
            action = select_action(state)
            print('les actions sont', action)
            if render:
                env.render()
            # print("env steop action", env.step(action))
            obs = env.step(action)[0]
            reward = env.step(action)[1]
            done = env.step(action)[2]

            total_reward += reward
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state
            if steps_done > INITIAL_MEMORY:
                loss = optimize_model()
                if loss is not None:
                    loss_values.append(loss)
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
        steps_done_.append(steps_done)
        episode_.append(episode)
        t_.append(t)
        total_reward_.append(total_reward)
        # if episode % 20 == 0:
                # print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t Loss value: {}'.format(steps_done, episode, t, total_reward, loss_values))
    env.close()

    with open('test.csv', 'w', newline='') as csvfile:
        fieldnames = ['episode', 'steps_done', 'timestep', 'total_reward', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(episode_)):
            loss = loss_values[i] if i < len(loss_values) else None
            writer.writerow({'episode': episode_[i], 'steps_done': steps_done_[i], 'timestep': t_[i], 'total_reward': total_reward_[i], 'loss': loss})
    
    return steps_done_, episode_, t_, total_reward_, loss_values

def test(env, n_episodes, policy, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state).max(1)[1].view(1,1)
            if render:
                env.render()
                time.sleep(0.02)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            state = next_state
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32 
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000 #etait a 1000000
    TARGET_UPDATE = 1000
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    # create networks
    model_path = "demonattack_model"
    if os.path.exists(model_path):
        policy_net = torch.load(model_path).to(device)
        print("Modèle chargé avec succès.")
    else:
        policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    env = gym.make("DemonAttack-v4")
    env = make_env(env)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    Train = train(env, 10)
    torch.save(policy_net, "demonattack_model")
    policy_net = torch.load("demonattack_model")