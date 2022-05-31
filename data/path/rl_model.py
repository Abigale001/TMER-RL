
######################
# Environment
EPI_REWARD = 100
DIM_ACTION = 1
DIM_STATE = 100
DIM_OBS = 4
END_RANGE = 0.005
STEP_RANGE = 0.05
MAX_STEPS = 200
FIG_FORMAT = '.png'
IMG = 'img/'
CKPT = 'ckpt/'
device = 'cpu'
ACTION_RANGE=5
######################
# Network

DIM_HIDDEN = 256
DIM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 2

######################
# Training
TRAIN = True
MAX_EPISODES = 20000
BATCH_SIZE  = 64
FIL_LR = 1e-3 
PLA_LR = 1e-3 
SAVE_ITER = 1000
SUMMARY_ITER = 1000
DISPLAY_ITER = 10
SHOW_TRAJ = True
SHOW_DISTR = False


PF_RESAMPLE_STEP = 3
NUM_PAR_PF = 100



NUM_PAR_SMC_INIT = 3
NUM_PAR_SMC = 30
HORIZON = 10
SMCP_MODE = 'topk' # 'samp', 'topk'
SMCP_RESAMPLE = True
SMCP_RESAMPLE_STEP = 3


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
replay_buffer_size = 10000
alpha = 1.0
gamma = 0.95
tau = 0.005
const = 1e-6

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.distributions.categorical import Categorical

#########################
# Training Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s_t, a_t, r_t, s_tp1, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s_t, a_t, r_t, s_tp1, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # batch=batch.detach().numpy()
        # for m in batch:


        state, action, reward, next_state, done = map(
            np.stack, zip(*batch))    #action 64,2
        # state=state.detach().numpy()
        # action=action.detach().numpy()
        # reward=reward.detach().numpy()
        # next_state=next_state.detach().numpy()
        # done=done.detach().numpy()
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

#########################
# Planning Network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):  #30,2
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    

    def __init__(self, DIM_STATE=100, DIM_ACTION=10, DIM_HIDDEN=256):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(100, 256)
        self.linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, 1)
        self.log_std_linear = nn.Linear(256, 1)       
        self.apply(weights_init_)

    def forward(self, state):
        state=state.squeeze()
        if type(state) is np.ndarray:
            state = torch.Tensor(state)
        x = F.relu(self.linear1(state))   #state  30*2    x 30*256
        x = F.relu(self.linear2(x))   #30*256
        mean = self.mean_linear(x)   #30,2
        log_std = self.log_std_linear(x)  #30,2
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):   #state   30,2  # normal sample

        mean, log_std = self.forward(state)  #all 30,2

        std = log_std.exp()  #30,2
        normal = Normal(mean, std)
        x_t = normal.rsample() # 30,2   # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t) #30,2
        log_prob = normal.log_prob(x_t) #30,2
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + const)  #30,2
        return action, log_prob, torch.tanh(mean)

    def get_action(self, state):
        a, log_prob, _ = self.sample(state)  #a 30,2   log 30,1   _ 30,2 
        a=a*ACTION_RANGE
        a=a.floor()
        a=a+ACTION_RANGE
        return a, log_prob

#########################
# Training Process
class SMCP:
    def __init__(self):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.MSE_criterion = nn.MSELoss()

        # Planning
        self.critic = QNetwork(DIM_STATE, DIM_ACTION, DIM_HIDDEN)
        self.critic_optim = Adam(self.critic.parameters(), lr=PLA_LR)
        self.critic_target = QNetwork(DIM_STATE, DIM_ACTION, DIM_HIDDEN)
        hard_update(self.critic_target, self.critic)
        self.target_entropy = -torch.prod(torch.Tensor(DIM_ACTION)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = Adam([self.log_alpha], lr=PLA_LR)
        self.policy = GaussianPolicy(DIM_STATE * (NUM_PAR_SMC + 1), DIM_ACTION, DIM_HIDDEN)
        self.policy_optim = Adam(self.policy.parameters(), lr=PLA_LR)
        self.buffer = ReplayMemory(replay_buffer_size)


    def save_model(self, path):
        stats = {}
        stats['p_net'] = self.policy.state_dict()
        stats['c_net'] = self.critic.state_dict()


        torch.save(stats, path)

    def load_model(self, path):
        stats = torch.load(path)



        # Planning
        self.policy.load_state_dict(stats['p_net'])
        self.critic.load_state_dict(stats['c_net'])



    def get_q(self, state, action):
        qf1, qf2 = self.critic(state, action)  #30,1   30,1
        q = torch.min(qf1, qf2)
        return q

    def full_state_smc_planning(model, state, goal, env):
        # with torch.no_grad():
        goal = goal.expand(NUM_PAR_SMC, -1)
        smc_state  = state.reshape(1, 1, -1).repeat(HORIZON, NUM_PAR_SMC, 1)
        smc_action = torch.zeros(HORIZON, NUM_PAR_SMC, DIM_ACTION).to(device)
        smc_weight = torch.zeros(NUM_PAR_SMC).to(device)
        prev_q     = 0
        prev_r     = 0 
        for i in range(HORIZON):
            s = smc_state[i]
            a, log_prob = model.policy.get_action(s)
            r,ns=env.step(a)

            q = model.get_q(s, a, goal).view(-1)
            advantage = prev_r + q - prev_q - log_prob.view(-1)
            smc_weight += advantage

            prev_r = r
            prev_q = q
            smc_action[i] = a
            if i < HORIZON - 1:
                smc_state[i+1] = ns.clone()

        normalized_smc_weight = F.softmax(smc_weight, -1)
        n = Categorical(normalized_smc_weight).sample()
        a = smc_action[0,n]
        return a


    def soft_q_update(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(BATCH_SIZE)
        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1) # (B, 1)
        mask_batch = torch.FloatTensor(np.float32(1 - done_batch)).unsqueeze(1)
        state_batch=state_batch.squeeze(1)
        done_batch=torch.Tensor(done_batch)

        # ------------------------
        #  Train SAC
        # ------------------------


   
        next_state_action, next_state_log_pi, _ = self.policy.sample(state_batch.view(BATCH_SIZE, -1))
        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss =qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()


        pi, log_pi, _ = self.policy.sample(state_batch.view(BATCH_SIZE, -1))
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        soft_update(self.critic_target, self.critic, self.tau)










