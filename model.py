
import torch   
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import random
import torch.optim as optim
from coor_env import CoorEnv   
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Train on device = {device}")
class AttentionComm(nn.Module):
    def __init__(self, obs_dim, hidden_dim_1=256, hidden_dim_2=128, msg_dim=64):
        super().__init__()
        self.msg_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, msg_dim)
        )
        self.query_layer = nn.Linear(obs_dim, msg_dim)
        self.key_layer = nn.Linear(obs_dim, msg_dim)
    
    def forward(self, obs_batch):
        # obs_batch: Tensor [batch_size, n_agents, obs_dim]
        # return: hidden_states [batch_size, n_agents, obs_dim + msg_dim]
        msg_vecs = self.msg_encoder(obs_batch) # [batch_size, n_agents, msg_dim]
        query_vecs = self.query_layer(obs_batch)
        key_vecs = self.key_layer(obs_batch) # [batch_size, n_agents, msg_dim]

        attn_logits = torch.matmul(query_vecs, key_vecs.transpose(1, 2)) # [batch_size, n_agents, n_agents]
        attn_weights = F.softmax(attn_logits, dim=-1) # [batch_size, n_agents, n_agents]
        msg_agg = torch.matmul(attn_weights, msg_vecs) # [batch_size, n_agents, msg_dim]

        return torch.cat([obs_batch, msg_agg], dim=-1) 

class ReplayBufferGRU:
    """
    Replay buffer for agent with GRU network
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, state, action, last_action, reward, next_state
        )
        self.position = int((self.position +1) % self.capacity)
    
    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst = [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            #h_in = [1, batchsize= 1 , n_agents, hidden_size]
            min_seq_len = min(len(state), min_seq_len)
            hi_lst.append(h_in)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()
        ho_lst = torch.cat(ho_lst, dim=-3).detach()

        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            sample_len = len(state)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx + min_seq_len
            s_lst.append(state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            la_lst.append(last_action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
        
        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst
    
    def ___len__(self):
        return len(self.buffer) 
    
    def get_length(self):
        return len(self.buffer)
            

class RNNAgent(nn.Module):
    '''
    @  This class evaluate the Q value given a state + action
    '''
    def  __init__(self, num_inputs, action_shape, num_actions, hidden_size):
        super(RNNAgent, self).__init__()
        self.num_inputs = num_inputs
        self.action_shape = action_shape
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs+action_shape*num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)   
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action_shape*num_actions)

    def forward(self, state, action, hidden_in):
        '''
        @params:
            state: [#B, #S, #Agent, n_feature] 
            action: [#B, #S, #Agent, action_shape]
        @return:
            qs: [#B, #S, #Agent, action_shape, num_actions]
        '''       
        # print("state.shape  = ", state.shape)
        # print("action = ", action)
        bs, seq_len, n_agents, _ = state.shape
        state = state.permute(1,0,2,3) # state = [#S, #B, #Agent, n_feature]
        action = action.permute(1,0,2,3) # action = [#S, #B, #Agent, n_feature]
        # print("action.shape in forward", action.shape)
        # print("action before one_hot:", action, action.dtype, action.shape)
        action = F.one_hot(action, num_classes=self.num_actions) # action = [#S, #B, #Agent, n_feature, num_actions]
        action = action.view(seq_len, bs, n_agents, -1)
        # -> action = [#S, #B, #Agent, n_feature * num_actions]
        x = torch.cat([state, action], -1) 
        # -> x = [#S, #B, #Agent, n_feature * num_actions + n_feature]
        x = x.view(seq_len, bs*n_agents, -1)
        hidden_in = hidden_in.view(1, bs*n_agents, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # -> x = [#S, #B, #Agent, hidden_size]
        # print("x.shape = ", x.shape)
        x, hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = self.linear4(x) 
        # [#S, #B, #agent, action_shape * num_actions]
        x = x.view(seq_len, bs, n_agents, self.action_shape, self.num_actions)
        qs =  F.softmax(x, dim = -1)
        qs =  qs.permute(1,0,2,3,4) # [#B, #S, #agent, #action_shape, #num_action]
        return qs, hidden
    
    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        for each distributed agent, generate action for one step given input data
        @params:
            state: [n_agents, n_feature]
            last_action : [n_agents, action_shape]
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        # Assuming last_action is the NumPy array [[6.], [115.], [75.], [219.]] with dtype float32
        # print("last_action in RNN ", last_action)
        last_action = torch.tensor(last_action, dtype=torch.long).to(device) 
        # print("last_action after from_numpy =", last_action, "dtype =", last_action.dtype)
        # Add batch and sequence dimensions
        last_action = last_action.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 4, 1)
        # print("last_action after unsqueeze =", last_action, "dtype =", last_action.dtype)
        hidden_in = hidden_in.unsqueeze(1)

        # print("abc_123")
        # print("last_action before forward =", last_action.tolist(), "dtype =", last_action.dtype, "shape =", last_action.shape)
        agent_outs, hidden_out = self.forward(state, last_action, hidden_in)
        # agent_outs = [#B, #S, #agent, #action_shape, #num_action]
        dist = Categorical(agent_outs)
        if deterministic:
            action = np.argmax(agent_outs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze(0).squeeze(0).detach().cpu().numpy()
        return action, hidden_out

class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, action_shape, embed_dim=64, hypernet_embed=128, abs=True):
        """
        Critic network of QMIX
        """
        super(QMix, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim*n_agents*action_shape
        self.action_shape = action_shape
        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.abs = abs
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(self.hypernet_embed, self.action_shape * self.embed_dim * self.n_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(self.hypernet_embed, self.embed_dim)
            )
        
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states):
        """
        Return q value for the given outputs
        @param
            agent_qs: [#B, #S, #Agent, #action_shape]
            states: [#B, #S, #Agent, #features*action_shape]
        : param agent_qs: q value inputs into network 
        : param states: state observation
        : return q_tot: q total
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim) 
        # [#batch * #sequence, action_shape * #feature * #agent]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents*self.action_shape) 
        # [#batch * #sequence, 1, #Agent * action_shape]
        # First layer of QMIX
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        # w1 = [#batch, #seq, action_shape * n_agents * self.embed_dim]
        b1 = self.hyper_b_1(states)
        # b1 = [#batch, #seq, self.embed_dim]
        w1 = w1.view(-1, self.n_agents*self.action_shape, self.embed_dim)
        # w1 = [#batch * #seq, n_agents *  action_shape, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)
        # b1 = [#batch * #seq, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # agent_qs = [#batch * #seq, 1, #agent * action_space]
        # w1 = [#batch * #seq, n_agents* action_space, embed_dim]
        # torch.bmm(agent_qs, w1) = [#batch * #seq, 1, embed_dim]
        # hidden = [#batch * #seq, 1, embed_dim]
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        # w_final = [#batch, #seq, self.embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1) 
        # w_final = [#batch * #seq, self.embed_dim, 1]
        v = self.V(states)
        # V  = [#batch, #seq, 1]
        v = v.view(-1, 1, 1)
        q_tot = F.elu(torch.bmm(hidden, w_final) + v)
        # hidden = [#batch * #seq, 1, embed_dim]
        # w_final = [#batch * #seq, self.embed_dim, 1]
        # torch.bmm(hidden, w_final) = [#batch * #seq, 1]
        # q_tot = [#batch * #seq, 1]
        q_tot = q_tot.view(bs, -1, 1)
        # q_tot = [#batch, #seq, 1]
        return q_tot

class QMix_Trainer(nn.Module):
    def __init__(self, replay_buffer, 
                 n_agents, state_dim, action_shape, action_dim, hidden_dim, msg_dim,
                 hypernet_dim, target_update_interval, 
                 lr=0.001, logger=None):
        super(QMix_Trainer, self).__init__()
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.target_update_interal = target_update_interval
        self.encode_dim = state_dim + msg_dim
        self.agent = RNNAgent(self.encode_dim, action_shape,
                              action_dim, hidden_dim).to(device)
        #action_dim = num_actions
        self.target_agent = RNNAgent(self.encode_dim, action_shape,
                                     action_dim, hidden_dim).to(device)
        self.mixer = QMix(self.encode_dim, n_agents, action_shape, hidden_dim, hypernet_dim).to(device)
        
        self.target_mixer = QMix(self.encode_dim, n_agents, action_shape,
                                 hidden_dim, hypernet_dim).to(device) 
        
        self.attn_comm = AttentionComm(obs_dim=state_dim, msg_dim=msg_dim)


        self._update_targets()
        self.update_cnt = 0
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            list(self.attn_comm.parameters()) + list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"👉 Tổng số tham số trainable: {total_params}")
    
    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim
        ).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.n_agents, self.action_shape))

        return action.type(torch.FloatTensor).numpy()
    
    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @return:
            action: w/ shape [#active_as]
        '''
        #state = [Batch, sequence, num_agents, state_dim]
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        b, s, n, f = state.shape
        state = state.view(b*s, n , f)
        state = self.attn_comm(state)
        state = state.squeeze(0).to(device)
        # print("state.shape in get action after attention = ", state.shape)
        action, hidden_out = self.agent.get_action(state, last_action, hidden_in, deterministic)
        return action, hidden_out
    
    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_state, episode_action,
                            episode_last_action, episode_reward, episode_next_state):
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action,
                                episode_last_action, episode_reward, episode_next_state)
    
    def update(self, batch_size):
        hidden_in, hidden_out, state, action, last_action, reward, next_state = self.replay_buffer.sample(
            batch_size
        )
        state = torch.FloatTensor(next_state).to(device)
        #state = [Batch, sequence, agents, features * action_shape]
        next_state = torch.FloatTensor(next_state).to(device)

        ####################################
        ## APPLIED ATTENTION COMMUNICATION##
        ####################################
        b, s, n, f = state.shape
        state = state.view(b*s, n, f)
        next_state = next_state.view(b*s, n, f)
        state = self.attn_comm(state)
        next_state = self.attn_comm(next_state)
        state = state.view(b, s, n, -1)
        next_state = next_state.view(b, s, n, -1)
        #####################################


        action = torch.LongTensor(action).to(device)
        #action = [Batch, sequence, agents, action_shape]
        last_action = torch.LongTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)
        #reward = [Batch, sequence] -> [Batch, sequence, 1]
        agent_outs, _ = self.agent(state, last_action, hidden_in)
        #agent_outs = [batch, sequence, n_agents, action_shape, action_dim]
        #action = [Batch, sequence, agents, action_shape]
        chosen_action_qvals = torch.gather(
            agent_outs, dim=-1, index=action.unsqueeze(-1)
        ).squeeze(-1)
        #action = [batch, sequence, agents, action_shape]
        #index = [Batch, sequence, agents, action_shape, 1]
        #->gather -> q_values for chosen action
        #chosen_action_qvals = [Batch, sequence, agents, action_shape, 1] 
        #q_vals = [Batch, sequence, agents, action_shape]

        qtot = self.mixer(chosen_action_qvals, state)
        #qtot = [Batch, sequence, 1]

        #target q
        target_agent_outs, _ = self.target_agent(next_state, action, hidden_out)
        #target_agent_outs = [batch, sequence, num_agents, action_shape, action_dim]
        # .max return : [0]: values, [1]: indices
        target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0]
        #target_max_qvals = [batch, sequence, num_agents, action_shape, 1]
        target_qtot = self.target_mixer(target_max_qvals, next_state)
        #target_qtot = [batch, sequence, 1]

        reward = reward[:,:,0]
        #reward : [Batch, sequence, 1]

        targets = self._build_td_lambda_targets(reward, target_qtot)
        loss = self.criterion(qtot, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt +=1
        if self.update_cnt % self.target_update_interal == 0:
            self._update_targets()
        
        return loss.item()

    def _build_td_lambda_targets(self, rewards, target_qs, gamma=0.99, td_lambda=0.6):
        '''
        @params:
            rewards: [#batch, #sequence, 1]
            target_qs: [#batch, #sequence, 1]
        '''
        # print("test reward.shape = ", rewards.shape)
        # print("test target_qs.shape = ", target_qs.shape)
        rewards = rewards.unsqueeze(-1)
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1]
        # backwards recursive update of the "forward view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1])
        return ret
    
    def _update_targets(self):
        #Update target networks
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, path):
        torch.save(self.agent.state_dict(), path+'_agent')
        torch.save(self.mixer.state_dict(), path+'_mixer')
        torch.save(self.attn_comm.state_dict(), path+'_att')

    def load_model(self, path):
        self.agent.load_state_dict(torch.load(path+'_agent'))
        self.mixer.load_state_dict(torch.load(path+'_mixer'))
        self.attn_comm.load_state_dict(torch.load(path+'_att'))

        self.agent.eval()
        self.mixer.eval()
        self.attn_comm.eval()
