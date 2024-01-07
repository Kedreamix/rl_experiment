import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import datetime
import gym

from wrappers import *
from memory import ReplayMemory
from models import *
from gym.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import argparse
Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

def optimize_model():
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
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
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

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    rewards = []  # 记录所有回合的奖励
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        # add tensorboard
        writer.add_scalar('reward', total_reward, episode)
        writer.add_scalar('Mean reward', np.mean(rewards[-10:]), episode)
        if episode % 10 == 0:
            print('Total steps: {} \t Episode: {}/{} Step: {} \t Total reward: {}'.format(steps_done, episode, n_episodes, t, total_reward))
            writer.add_scalar('Mean reward', np.mean(rewards[-10:]), episode)
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    #  env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    env = RecordVideo(env, './videos/' + ENV_NAME)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

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


def get_args():
    """ 
    超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm") 
    parser.add_argument('--env_name',default='PongNoFrameskip-v4',type=str,help="name of environment") # PongNoFrameskip, SpaceInvadersNoFrameskip
    parser.add_argument('--train_eps',default=400,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=5,type=int,help="episodes of testing")
    parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.5,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=1000000,type=int,help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--target_update',default=1000,type=int)
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--seed',default=10,type=int,help="seed") 
    parser.add_argument('--render', action='store_true', help='render the environment')
    args = parser.parse_args([])
    args = {**vars(args)}  # 转换成字典类型    
    ## 打印超参数
    print("超参数")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))      
    return args

if __name__ == '__main__':
    cfg = get_args() 
    # set device
    device = cfg['device']

    # hyperparameters
    ALGO_NAME = cfg['algo_name']
    BATCH_SIZE = cfg['batch_size']
    GAMMA = cfg['gamma']
    EPS_START = cfg['epsilon_start']
    EPS_END = cfg['epsilon_end']
    EPS_DECAY = cfg['epsilon_decay']
    TARGET_UPDATE = cfg['target_update']
    RENDER = cfg['render']
    lr = cfg['lr']
    INITIAL_MEMORY = 10000
    # MEMORY_SIZE = 10 * INITIAL_MEMORY
    MEMORY_SIZE = cfg['memory_capacity']
    ENV_NAME = cfg['env_name']
    RENDER = True
    
    # create networks
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    steps_done = 0
    
    # create environment
    env = gym.make(ENV_NAME)
    env = make_env(env)
    
    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # tensorboard
    writer = SummaryWriter(f'runs/{ALGO_NAME}_{ENV_NAME}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    
    # train model
    train(env, cfg['train_eps'])
    torch.save(policy_net, f"{ALGO_NAME}_{ENV_NAME}_model.pth")
    policy_net = torch.load( f"{ALGO_NAME}_{ENV_NAME}_model.pth")
    test(env, cfg['test_eps'], policy_net, render=False)

##########################################################################
    # TARGET_UPDATE = 2000
    # BATCH_SIZE = 64
    # GAMMA = 0.99
    # print("BATCH_SIZE = 64 down")  # 打印结束信息

    # # create networks
    # policy_net = DQN(n_actions=4).to(device)
    # target_net = DQN(n_actions=4).to(device)
    # target_net.load_state_dict(policy_net.state_dict())

    # # setup optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # steps_done = 0

    # # create environment
    # env = gym.make("PongNoFrameskip-v4")
    # env = make_env(env)

    # # initialize replay memory
    # memory = ReplayMemory(MEMORY_SIZE)

    # # train model
    # train(env, 400)
    # torch.save(policy_net, "dqn_pong_model")
    # policy_net = torch.load("dqn_pong_model")
    # test(env, 1, policy_net, render=False)

    ##########################################################################
    # TARGET_UPDATE = 1000
    # BATCH_SIZE = 64
    # GAMMA = 0.99
    # print("BATCH_SIZE = 64 down")  # 打印结束信息

    # # create networks
    # policy_net = DQN(n_actions=4).to(device)
    # target_net = DQN(n_actions=4).to(device)
    # target_net.load_state_dict(policy_net.state_dict())

    # # setup optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # steps_done = 0

    # # create environment
    # env = gym.make("PongNoFrameskip-v4")
    # env = make_env(env)

    # # initialize replay memory
    # memory = ReplayMemory(MEMORY_SIZE)

    # # train model
    # train(env, 400)
    # torch.save(policy_net, "dqn_pong_model")
    # policy_net = torch.load("dqn_pong_model")
    # test(env, 1, policy_net, render=False)

#
# ##########################################################################
    # TARGET_UPDATE = 1000
    # BATCH_SIZE = 128
    # GAMMA = 0.99
    # print("BATCH_SIZE = 128 down")  # 打印结束信息

    # # create networks
    # policy_net = DQN(n_actions=4).to(device)
    # target_net = DQN(n_actions=4).to(device)
    # target_net.load_state_dict(policy_net.state_dict())

    # # setup optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # steps_done = 0

    # # create environment
    # env = gym.make("PongNoFrameskip-v4")
    # env = make_env(env)

    # # initialize replay memory
    # memory = ReplayMemory(MEMORY_SIZE)

    # # train model
    # train(env, 400)
    # torch.save(policy_net, "dqn_pong_model")
    # policy_net = torch.load("dqn_pong_model")
    # test(env, 1, policy_net, render=False)

#
#
# ##########################################################################
#     TARGET_UPDATE = 1000
#     BATCH_SIZE = 32
#     GAMMA = 0.9
#     print("GAMMA = 0.9 down")  # 打印结束信息
#
#     # create networks
#     policy_net = DQN(n_actions=4).to(device)
#     target_net = DQN(n_actions=4).to(device)
#     target_net.load_state_dict(policy_net.state_dict())
#
#     # setup optimizer
#     optimizer = optim.Adam(policy_net.parameters(), lr=lr)
#
#     steps_done = 0
#
#     # create environment
#     env = gym.make("PongNoFrameskip-v4")
#     env = make_env(env)
#
#     # initialize replay memory
#     memory = ReplayMemory(MEMORY_SIZE)
#
#     # train model
#     train(env, 400)
#     torch.save(policy_net, "dqn_pong_model")
#     policy_net = torch.load("dqn_pong_model")
#     test(env, 1, policy_net, render=False)
#
#
#
# ##########################################################################
#     TARGET_UPDATE = 1000
#     BATCH_SIZE = 32
#     GAMMA = 0.8
#     print("GAMMA = 0.8 down")  # 打印结束信息
#
#     # create networks
#     policy_net = DQN(n_actions=4).to(device)
#     target_net = DQN(n_actions=4).to(device)
#     target_net.load_state_dict(policy_net.state_dict())
#
#     # setup optimizer
#     optimizer = optim.Adam(policy_net.parameters(), lr=lr)
#
#     steps_done = 0
#
#     # create environment
#     env = gym.make("PongNoFrameskip-v4")
#     env = make_env(env)
#
#     # initialize replay memory
#     memory = ReplayMemory(MEMORY_SIZE)
#
#     # train model
#     train(env, 400)
#     torch.save(policy_net, "dqn_pong_model")
#     policy_net = torch.load("dqn_pong_model")
#     test(env, 1, policy_net, render=False)
#
