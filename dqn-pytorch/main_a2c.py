import time
import os
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
import warnings
warnings.filterwarnings('ignore')

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def all_seed(env, seed = 1):
    ''' 万能的seed函数
    '''
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(env, n_episodes, render=False):
    total_rewards = []
    for episode in range(n_episodes):
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            # action = select_action(state)
            dist, value = a2c_net(state)

            action = dist.sample() # 分布采样
            
            if render:
                env.render()

            obs, reward, done, info = env.step(action.detach().cpu().numpy())
            
            log_prob = dist.log_prob(action) # 计算log概率
            entropy += dist.entropy().mean() # 计算熵
            
            log_probs.append(log_prob) # 记录log概率
            values.append(value) # 记录value
            
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                break
        
        _, next_value = a2c_net(next_state)
        returns = compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        loss = actor_loss + critic_loss - 0.001 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_rewards.append(total_reward)
        # add tensorboard
        writer.add_scalar('reward', total_reward, episode)
        writer.add_scalar('epsilon', epsilon, episode)
        if episode % 10 == 0:
            print('Step/Total steps: {}/{} \t Episode: {}/{} \t Total/Mean reward: {}/{:.1f} \t Epislon: {:.2f}'.format(
                t, steps_done, episode, n_episodes, total_reward, np.mean(total_rewards[-10:]), epsilon))
            writer.add_scalar('Mean reward', np.mean(total_rewards[-10:]), episode)
            torch.save(a2c_net, save_path)
    env.close()
    return

def test(env, n_episodes, a2c_net, render=True):
    #  env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    env = RecordVideo(env, './videos/' + ENV_NAME)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            dist, value = a2c_net(state)

            action = dist.sample() # 分布采样

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action.detach().cpu().numpy())

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
    parser.add_argument('--algo_name',default='A2C',type=str,help="name of algorithm") 
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
    args = parser.parse_args()
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
    SEED = cfg['seed']
    RENDER = True

    steps_done = 0
    epsilon = EPS_START
    
    # create environment
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    env = make_env(env)
    if SEED != 0:
        all_seed(env, SEED)

    N_ACTIONS = env.action_space.n
    # create networks
    a2c_net = ActorCritic(n_actions=N_ACTIONS).to(device)
    
    # setup optimizer
    optimizer = optim.Adam(a2c_net.parameters(), lr=lr)
    
    # tensorboard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_path = f'runs/{ALGO_NAME}_{ENV_NAME}{timestamp}'
    writer = SummaryWriter(tensorboard_path)
    
    start = time.time()
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/{ALGO_NAME}_{ENV_NAME}_{timestamp}_model.pth"
    
    # train model
    train(env, cfg['train_eps'])
    print('Tensorboard command: tensorboard --logdir=' + tensorboard_path)
    
    a2c_net = torch.load( save_path)
    test(env, cfg['test_eps'], a2c_net, render=False)
    print("Using time {} hours".format((time.time()-start)/3600))