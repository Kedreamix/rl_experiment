import gym
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import cv2
from collections import deque
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from torch.utils.tensorboard import SummaryWriter   
warnings.filterwarnings("ignore")
from gym.wrappers import AtariPreprocessing, LazyFrames, FrameStack
class DQNModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim = 512, in_channels = 1):
        """ 
        初始化q网络
        """
        super(DQNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
    
class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        ''' 
        存储transition到经验回放中
        '''
        self.buffer.append(transitions)
        
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer): # 如果批量大小大于经验回放的容量，则取经验回放的容量
            batch_size = len(self.buffer)
        if sequential: # 顺序采样
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else: # 随机采样
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        ''' 
        清空经验回放
        '''
        self.buffer.clear()
    def __len__(self):
        ''' 
        返回当前存储的量
        '''
        return len(self.buffer)
    
    
class DQN:
    def __init__(self,model,memory,cfg):
        self.n_actions = cfg['n_actions']  
        self.device = torch.device(cfg['device']) 
        self.gamma = cfg['gamma'] # 奖励的折扣因子
        # e-greedy策略相关参数
        self.sample_count = 0  # 用于epsilon的衰减计数
        self.epsilon = cfg['epsilon_start']
        self.sample_count = 0  
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        # 复制参数到目标网络
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg['lr']) # 优化器
        self.memory = memory # 经验回放
    
    def sample_action(self, state):
        ''' 
        采样动作
        '''
        self.sample_count += 1
        # epsilon指数衰减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                # state = prepro(state)
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)# 因为默认没有0
        return action
    
    @torch.no_grad() # 不计算梯度，该装饰器效果等同于with torch.no_grad()：
    def predict_action(self, state):
        ''' 
        预测动作
        '''
        # state = prepro(state)
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
    
    def update(self):
        if len(self.memory) < self.batch_size: # 当经验回放中不满足一个批量时，不更新策略
            return
        # 从经验回放中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 将数据转换为tensor
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()  
        loss.backward()
        # clip防止梯度爆炸
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 


def prepro(I):
    """prepro 210x160x3 frame into 168x168, then downscale to 84x84 1D float vector"""
    I = I[27:195] # crop to 168x160
    I = I[:,:,0]  # Convert to grayscale by taking one channel

    # Add 4 black pixels on each side to make it 168x168
    I = cv2.copyMakeBorder(I, 0, 0, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # Downscale to 84x84
    I = cv2.resize(I, (84, 84), interpolation=cv2.INTER_AREA)

    I[I == 0] = 0 # You may need to change this value based on the actual background color in Space Invaders
    I[I != 0] = 1  # Set all other (non-zero) pixels to 1

    return I.astype(np.float).ravel().reshape(1, 84, 84)  
        
def train(cfg, env, agent):
    ''' 
    训练
    '''
    
    print("开始训练！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    start_eps = 0
    ckpt_path = f'{cfg["env_name"]}_{cfg["algo_name"]}.pth'
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path)
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rewards = checkpoint['rewards']
            steps = checkpoint['steps']
            start_eps = checkpoint['start_eps']
            print(f"加载上一次的训练，起始回合为{start_eps}")
        except:
            print("加载上一次的训练失败，重新开始")
    
    for i_ep in range(start_eps, cfg['train_eps']):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        prev_input = None 
        for _ in range(cfg['ep_max_steps']):
            ep_step += 1
            
            action = agent.sample_action(state)  # 选择动作
            if cfg['n_actions'] == 3 and cfg['env_name'] == 'SpaceInvadersNoFrameskip-v4':
                next_state, reward, done, _ = env.step(action+1)  # 更新环境，返回transition
            else:
                next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push((state, action, reward,next_state, done)) # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        
        if (i_ep + 1) % cfg['target_update'] == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep+1}/{cfg['train_eps']}，奖励：{ep_reward:.2f}，Epislon：{agent.epsilon:.3f}")
            # 保存权重
            checkpoint = {
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'rewards': rewards,
                'steps': steps,
                'start_eps': i_ep
            } 
            torch.save(checkpoint, ckpt_path)
            writer.add_scalar('Mean reward', np.mean(rewards[-10:]), i_ep)
            
    print("完成训练！")
    env.close()
    return {'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg['test_eps']):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg['ep_max_steps']):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            if cfg['n_actions'] == 3 and cfg['env_name'] == 'SpaceInvadersNoFrameskip-v4':
                next_state, reward, done, _ = env.step(action+1)  # 更新环境，返回transition
            else:
                next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    print("完成测试")
    env.close()
    return {'rewards':rewards}

def all_seed(env,seed = 1):
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
    
def env_agent_config(cfg):
    env = gym.make(cfg['env_name']) # 创建环境
    env = AtariPreprocessing(env,
                         scale_obs=False,
                         terminal_on_life_loss=True,
                         )
    env = FrameStack(env, num_stack=4)
    if cfg['seed'] !=0:
        all_seed(env,seed=cfg['seed'])
    # n_states = env.observation_space.shape[0]
    # n_actions = env.action_space.n
    n_actions = env.action_space.n
    n_states = env.observation_space.shape
    n_actions = min(n_actions, cfg['n_actions'])
    # FIRE_ACTION = 1
    # RIGHT_ACTION = 2
    # LEFT_ACTION = 3
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    cfg.update({"n_states":n_states,"n_actions":n_actions}) # 更新n_states和n_actions到cfg参数中
    model = DQNModel(n_states, n_actions, hidden_dim = cfg['hidden_dim'], in_channels = cfg['in_channels']) # 创建模型
    memory = ReplayBuffer(cfg['memory_capacity'])
    agent = DQN(model,memory,cfg)
    return env,agent


def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()
    plt.savefig(f"{cfg['algo_name']}_{cfg['env_name']}_{tag}.png")

def get_args():
    """ 
    超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm") 
    parser.add_argument('--env_name',default='PongNoFrameskip-v4',type=str,help="name of environment") # PongNoFrameskip-v4, SpaceInvadersNoFrameskip-v4
    parser.add_argument('--train_eps',default=400,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.5,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=500,type=int,help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--target_update',default=1000,type=int)
    parser.add_argument('--hidden_dim',default=512,type=int)
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--seed',default=10,type=int,help="seed") 
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--in_channels', default=4, type=int)
    parser.add_argument('--n_actions', default=100, type=int)
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

if __name__ == "__main__":
    # 获取参数
    cfg = get_args() 
    writer = SummaryWriter('runs/{}_{}'.format(cfg['algo_name'] + '_' + cfg['env_name'] ,
                                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                               ))
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    
    plot_rewards(res_dic['rewards'], cfg, tag="train")  
    # 测试
    res_dic = test(cfg, env, agent)
    plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果