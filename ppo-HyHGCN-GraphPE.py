import csv
import os
import random
import statistics
import torch
import torch.nn.functional as F
import numpy as np
import gym
from torch import nn
from torch_geometric.nn import HypergraphConv, GraphConv

import rl_utils
from tqdm import tqdm
import __init__
import matplotlib.pyplot as plt
import pandas as pd


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


                                                              
      
             
                                                              
hyperedges = [
    [0, 1, 2],
    [3, 4, 10],
    [5, 6, 4],
    [7, 8, 29],
    [8],
    [11, 13, 15, 10, 1, 2, 6, 116],
    [14, 12, 13, 16, 18, 32],
    [17, 18, 16, 19, 20],
    [18, 17, 14],
    [23, 69, 71, 22],
    [26, 25, 21],
    [29, 24],
    [26, 24, 27, 31, 114],
    [30, 31, 16, 28],
    [31, 112, 113],
    [33, 18, 35, 36, 42],
    [35, 34, 33],
    [39, 36, 41, 40, 38],
    [41, 39, 40, 48],
    [45, 44, 46, 47],
    [48, 46, 41, 44, 47, 49, 50, 53, 65, 68],
    [53, 52, 48, 54, 55, 58, 51],
    [54, 53, 55, 58],
    [55, 53, 54, 56, 57, 58],
    [62, 63, 58],
    [62, 64, 60],
    [61, 59, 60, 65, 66],
    [64, 37, 63, 67, 65],
    [64, 48, 61, 66],
    [64, 80, 115, 68],
    [69, 68, 23, 70, 73, 74],
    [71, 23, 70],
    [72, 70],
    [73, 69, 74],
    [75, 76, 117],
    [76, 75, 68, 74, 77, 79, 81, 78],
    [80, 67, 79, 96],
    [84, 82, 83, 85, 87, 88],
    [85],
    [84, 87, 89, 91],
    [89, 88, 90],
    [90, 89, 91, 94, 95],
    [91, 88, 90, 92, 93, 99, 101],
    [98, 79, 99],
    [99, 91, 93, 97, 98, 100, 102, 103, 105],
    [102, 99, 103, 104, 109],
    [103, 99, 102, 104],
    [104, 102, 103, 105, 106, 107],
    [106, 104, 105],
    [109, 102, 108, 110, 111],
    [110, 109],
    [111, 109],
    [112, 16, 31],
    [115, 67]
]


def build_hyperedge_index(hyperedges):
    node_idx, edge_idx = [], []

    for e_id, nodes in enumerate(hyperedges):
        for n in nodes:
            node_idx.append(n)
            edge_idx.append(e_id)

    return torch.tensor([node_idx, edge_idx], dtype=torch.long)


def build_graph_laplacian_pe(edge_index, num_nodes, pe_dim, device):

                                         
    edge_index_cpu = edge_index.detach().cpu()

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    row, col = edge_index_cpu[0], edge_index_cpu[1]
    A[row, col] = 1.0

            
    A = torch.maximum(A, A.T)

         
    deg = A.sum(dim=1)

                
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

                                            
    I = torch.eye(num_nodes, dtype=torch.float32)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt

          
    eigvals, eigvecs = torch.linalg.eigh(L)

                             
    max_pe_dim = min(pe_dim, num_nodes - 1)
    pe = eigvecs[:, 1:max_pe_dim + 1]

                                    
    if max_pe_dim < pe_dim:
        pad = torch.zeros((num_nodes, pe_dim - max_pe_dim))
        pe = torch.cat([pe, pad], dim=1)

    return pe.to(device)


class HGNNAgent1(nn.Module):

    def __init__(self, hyperedge_index, edge_index, num_hyperedges, pos_enc=None):
        super(HGNNAgent1, self).__init__()

        self.hyperedge_index = hyperedge_index
        self.edge_index = edge_index
        self.num_hyperedges = num_hyperedges

        self.num_gen = 72
        self.hidden_channels = 16
        self.gamma = 0.99
        self.activation = torch.tanh
        self.n_out = 2 * self.num_gen
        self.obs_size = 118
        self.num_nodes = 118

                          
        if pos_enc is not None:
            self.register_buffer("pos_enc", pos_enc.float())
            self.pe_dim = pos_enc.shape[1]
        else:
            self.pos_enc = None
            self.pe_dim = 0

                                       
        self.hconv1 = HypergraphConv(1 + self.pe_dim, self.hidden_channels)

                                                  
        self.gconv1 = GraphConv(self.hidden_channels, 1)

                                       
        self.fc = nn.Linear(self.num_nodes, self.n_out)

    def _hybrid_conv(self, x_2d):

        hei = self.hyperedge_index.to(x_2d.device)
        ei = self.edge_index.to(x_2d.device)

                   
        if self.pos_enc is not None:
            pe = self.pos_enc.to(x_2d.device)
            x_2d = torch.cat([x_2d, pe], dim=1)

                         
        x = self.hconv1(x_2d, hei)
        x = torch.sigmoid(x)

                               
        x = self.gconv1(x, ei)
        x = torch.sigmoid(x)

        return x.reshape(self.num_nodes)

    def forward(self, obs, hyperedge_index=None):

        obs = obs.reshape(-1, self.num_nodes, 1)
        x = torch.as_tensor(obs).float().to(self.fc.weight.device)

        if x.shape[0] == 1:
            x = self._hybrid_conv(x[0]).unsqueeze(0)
        else:
            x = torch.stack([self._hybrid_conv(x[i]) for i in range(x.shape[0])])

        x = self.fc(x)
        x = x.view(-1, 2)
        x = F.softmax(x, dim=1)

        return x


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,
                 hyperedge_index, edge_index, num_hyperedges, pos_enc=None):

        self.hyperedge_index = hyperedge_index
        self.edge_index = edge_index

        self.actor = HGNNAgent1(
            self.hyperedge_index,
            self.edge_index,
            num_hyperedges,
            pos_enc=pos_enc
        ).to(device)

        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)

        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        return action.cpu().numpy()

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'],
            dtype=torch.float
        ).to(self.device)

        actions = torch.tensor(
            transition_dict['actions']
        ).view(-1, 1).to(self.device)

        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float
        ).to(self.device)

        rewards = rewards.reshape(-1, 1)

        next_states = torch.tensor(
            transition_dict['next_states'],
            dtype=torch.float
        ).to(self.device)

        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float
        ).to(self.device)

        dones1 = 1 - dones
        dones1 = dones1.view(-1, 1)

        td_target = rewards + self.gamma * self.critic(next_states) * dones1
        td_delta = td_target - self.critic(states)

        advantage = rl_utils.compute_advantage(
            self.gamma,
            self.lmbda,
            td_delta.cpu()
        ).to(self.device)

        epsilon = 1e-8

        old_log_probs = torch.log(
            self.actor(states).gather(1, actions) + epsilon
        ).detach()

                              
                          
        old_log_probs = old_log_probs.reshape(288, 72)

        for _ in range(self.epochs):
            log_probs = torch.log(
                self.actor(states).gather(1, actions) + epsilon
            )

            log_probs = log_probs.reshape(288, 72)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - self.eps,
                1 + self.eps
            ) * advantage

            sum54 = -torch.mean(torch.min(surr1, surr2), dim=1)
            actor_loss = torch.mean(sum54)

            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_models(self, path='ppo_hyhgcn_graphpe_checkpoint.pth'):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

        print(f"Checkpoint saved to {path}")

    def load_models(self, path='ppo_hyhgcn_graphpe_checkpoint.pth'):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(f"Checkpoint loaded from {path}")


def get_all_state(state):
    day_info = {
        key: state[key]
        for key in ['day', 'status', 'power', 'ens', 'cons', 'day_cost', 'timestep']
        if key in state
    }

    insert_into_csv(day_info)


def insert_into_csv(data):
    with open('day_cost_IES_HyHGCN_GraphPE1121.csv', 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=data.keys())

        csvfile.seek(0, 2)

        if csvfile.tell() == 0:
            csv_writer.writeheader()

        csv_writer.writerow(data)


def process_observation(state):
    obs_new_load = state['load']
    return obs_new_load


                                                              
      
                                                              
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

fix_seed(996)

actor_lr = 1e-3
critic_lr = 1e-2
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
num_episodes = 500

env_name = "UnitCommitmentEnv-v0"
env = gym.make(env_name)

status_space = env.observation_space.spaces['status']

state_dim = 118
action_dim = env.action_space.shape[0]

                                                              
      
                                                              
edgeindex = [
    [0, 1], [0, 2], [3, 4], [2, 4], [4, 5], [5, 6], [7, 8], [8, 9],
    [3, 10], [4, 10], [10, 11], [1, 11], [2, 11], [6, 11], [10, 12],
    [11, 13], [12, 14], [13, 14], [11, 15], [14, 16], [15, 16], [16, 17],
    [17, 18], [18, 19], [14, 18], [19, 20], [20, 21], [21, 22], [22, 23],
    [22, 24], [24, 26], [26, 27], [27, 28], [7, 29], [25, 29], [16, 30],
    [28, 30], [22, 31], [30, 31], [26, 31], [14, 32], [18, 33], [34, 35],
    [34, 36], [32, 36], [33, 35], [33, 36], [36, 38], [36, 39], [29, 37],
    [38, 39], [39, 40], [39, 41], [40, 41], [42, 43], [33, 42], [43, 44],
    [44, 45], [45, 46], [45, 47], [46, 48], [41, 48], [41, 48], [44, 48],
    [47, 48], [48, 49], [48, 50], [50, 51], [51, 52], [52, 53], [48, 53],
    [48, 53], [53, 54], [53, 55], [54, 55], [55, 56], [49, 56], [55, 57],
    [50, 57], [53, 58], [55, 58], [55, 58], [54, 58], [58, 59], [58, 60],
    [59, 60], [59, 61], [60, 61], [62, 63], [37, 64], [63, 64], [48, 65],
    [48, 65], [61, 65], [61, 66], [65, 66], [64, 67], [46, 68], [48, 68],
    [68, 69], [23, 69], [69, 70], [23, 71], [70, 71], [70, 72], [69, 73],
    [69, 74], [68, 74], [73, 74], [75, 76], [68, 76], [74, 76], [76, 77],
    [77, 78], [76, 79], [76, 79], [78, 79], [67, 80], [76, 81], [81, 82],
    [82, 83], [82, 84], [83, 84], [84, 85], [85, 86], [84, 87], [84, 88],
    [87, 88], [88, 89], [88, 89], [89, 90], [88, 91], [88, 91], [90, 91],
    [91, 92], [91, 93], [92, 93], [93, 94], [79, 95], [81, 95], [93, 95],
    [79, 96], [79, 97], [79, 98], [91, 99], [93, 99], [94, 95], [95, 96],
    [97, 99], [98, 99], [99, 100], [91, 101], [100, 101], [99, 102], [99, 103],
    [102, 103], [102, 104], [99, 105], [103, 104], [104, 105], [104, 106],
    [104, 107], [105, 106], [107, 108], [102, 109], [108, 109], [109, 110],
    [109, 111], [16, 112], [31, 112], [31, 113], [26, 114], [113, 114], [67, 115],
    [11, 116], [74, 117], [75, 117], [25, 24], [29, 16], [37, 36], [62, 58],
    [63, 60], [64, 65], [67, 68], [80, 79]
]

                                                              
        
                                                              
if len(hyperedges) == 0:
    print('提示：当前未输入超边数据，临时将普通边作为二节点超边使用。')
    hyperedges_used = [list(e) for e in edgeindex]
else:
    hyperedges_used = hyperedges

num_hyperedges = len(hyperedges_used)
hyperedge_index = build_hyperedge_index(hyperedges_used).to(device)

print(f'超边数量: {num_hyperedges}, hyperedge_index 形状: {tuple(hyperedge_index.shape)}')

                                                              
                        
                                                              
ei = np.array(edgeindex).transpose()

edge_index = torch.from_numpy(
    np.concatenate([ei, ei[::-1, :]], axis=1).copy()
).long().to(device)

print(f'普通边数量(含反向): {edge_index.shape[1]}')

                                                              
                                   
                            
                                                              
pe_dim = 4

pos_enc = build_graph_laplacian_pe(
    edge_index=edge_index,
    num_nodes=118,
    pe_dim=pe_dim,
    device=device
)

print(f'普通图 Laplacian PE 形状: {tuple(pos_enc.shape)}')

                                                              
              
                                                              
agent = PPO(
    state_dim,
    hidden_dim,
    action_dim,
    actor_lr,
    critic_lr,
    lmbda,
    epochs,
    eps,
    gamma,
    device,
    hyperedge_index,
    edge_index,
    num_hyperedges,
    pos_enc=pos_enc
)

return_list = []
cost_list = []
date_list = []

                                                              
      
                                                              
for i in tqdm(range(num_episodes), desc='Training Progress'):
    episode_return = 0

    state = env.reset()
    obs = process_observation(state)

    done = False

    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }

    while not done:
        obs = np.array(obs).astype(np.float32)

        action = agent.take_action(obs)

        next_state, reward, done, _ = env.step(action)

        get_all_state(next_state)

        if next_state['timestep'] == 287:
            cost_list.append(next_state['day_cost'])
            date_list.append(next_state['day'])

        next_obs = process_observation(next_state)

        transition_dict['states'].append(obs)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_obs)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)

        obs = next_obs

        episode_return += reward

    return_list.append(episode_return)

    agent.update(transition_dict)

                                                              
         
                                                              
agent.save_models('ppo_hyhgcn_graphpe1121.pth')

print('return', return_list)
print('cost', cost_list)

cost_list1 = cost_list[400:]

if len(cost_list1) > 0:
    print('average cost', statistics.fmean(cost_list1))

with open('return_cost_HyHGCN_GraphPE1121.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['day', 'cost', 'return'])

    for item1, item2, item3 in zip(date_list, cost_list, return_list):
        writer.writerow([item1, item2, item3])

print("数据已成功写入文件")

                                                              
        
                                                              
fig, ax = plt.subplots()

ax.plot(pd.Series(return_list).rolling(10).mean())

ax.set_title('PPO Performance on UC with HyHGCN + Graph Laplacian PE')
ax.set_xlabel('Episode')
ax.set_ylabel('Average Reward')

plt.tight_layout()
plt.show()
