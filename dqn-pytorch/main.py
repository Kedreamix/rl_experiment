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

def select_action(state):
    global steps_done
    global epsilon
    # 生成一个0-1之间的随机数
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    # epsilon = EPS_END + (EPS_START - EPS_END)* \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    epsilon = EPS_START - (EPS_START - EPS_END) * steps_done / LINEAR_DECAY_FRAMES
    if steps_done > LINEAR_DECAY_FRAMES:
        epsilon = EPS_END
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)

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
        rewards.append(total_reward)
        # add tensorboard
        writer.add_scalar('reward', total_reward, episode)
        writer.add_scalar('epsilon', epsilon, episode)
        if episode % 10 == 0:
            print('Step/Total steps: {}/{} \t Episode: {}/{} \t Total/Mean reward: {}/{:.1f} \t Epislon: {:.2f}'.format(
                t, steps_done, episode, n_episodes, total_reward, np.mean(rewards[-10:]), epsilon))
            writer.add_scalar('Mean reward', np.mean(rewards[-10:]), episode)
            torch.save(policy_net, save_path)
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    #  env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    video_dir = './videos/' + f'{ALGO_NAME}_{ENV_NAME}_{timestamp}'
    env = RecordVideo(env, video_dir, episode_trigger = lambda x: x < 3)
    for episode in range(n_episodes):
        # env = RecordVideo(env, video_dir + f'/{episode}')
        # print(video_dir + f'_{episode}')
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
    print("Video saved in {}".format(video_dir))
    env.close()
    return

def get_args():
    """ 
    超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm") 
    parser.add_argument('--env_name',default='PongNoFrameskip-v4',type=str,help="name of environment") # PongNoFrameskip, SpaceInvadersNoFrameskip
    parser.add_argument('--train_eps',default=1000,type=int,help="episodes of training")
    parser.add_argument('--ep_life', action='store_true', help='ep_life')
    parser.add_argument('--test_eps',default=3,type=int,help="episodes of testing")
    parser.add_argument('--ep_max_steps',default=100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.99,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.1,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=1000000,type=int,help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--target_update',default=1000,type=int)
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--seed',default=10,type=int,help="seed") 
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--test', default='', help='resume from checkpoint')
    parser.add_argument('--in_channels', default=4, type = int, help='input channels')
    parser.add_argument('--log', default='', help='log to tensorboard')
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
    TEST = cfg['test']
    EP_LIFE = cfg['ep_life']
    LINEAR_DECAY_FRAMES = 1000000
    RENDER = True

    steps_done = 0
    epsilon = EPS_START
    
    # create environment
    env = gym.make(ENV_NAME, 
                   render_mode='rgb_array')
    env = make_env(env, episodic_life = EP_LIFE, k = cfg['in_channels'])
    if SEED != 0:
        all_seed(env, SEED)

    N_ACTIONS = env.action_space.n
    # create networks
    policy_net = DQN(in_channels=cfg['in_channels'], n_actions=N_ACTIONS).to(device)
    target_net = DQN(in_channels=cfg['in_channels'], n_actions=N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.load_state_dict(policy_net.state_dict())
    # tensorboard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    start = time.time()
    
    if not TEST:   
        save_name =  f'{ALGO_NAME}_{ENV_NAME}_{timestamp}'
        if not cfg['log'] == '':
            save_name = cfg['log']
        
        # setup optimizer
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        
        # initialize replay memory
        memory = ReplayMemory(MEMORY_SIZE)

        tensorboard_path = f'runs/{save_name}'
        writer = SummaryWriter(tensorboard_path)

        os.makedirs('checkpoints', exist_ok=True)
        save_path = f"checkpoints/{save_name}_model.pth"
        
        # train model
        train(env, cfg['train_eps'])
        print('Tensorboard command: tensorboard --logdir=' + tensorboard_path)
    else:
        save_path = TEST
    policy_net = torch.load(save_path)
    test(env, cfg['test_eps'], policy_net, render=RENDER)
    print("Using time {} hours".format((time.time()-start)/3600))