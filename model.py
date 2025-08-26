import torch  
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import random
import torch.optim as optim
from coor_env import CoorEnv  
from torch.cuda.amp import GradScaler, autocast
import time
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
 
print(f"Train on device = {device}")
 
class AttentionComm_v2(nn.Module):
    def __init__(self, obs_dim, hidden_dim_1=512, msg_dim=256, att_msg_dim=512, num_heads=8):
        super().__init__()
        assert att_msg_dim % num_heads == 0, "att_msg_dim phải chia hết cho num_heads"
        self.num_heads = num_heads
        self.head_dim = att_msg_dim // num_heads
        self.msg_dim = msg_dim
        self.att_msg_dim = att_msg_dim
        self.msg_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, msg_dim),
            nn.ReLU()
        )
 
        # Attention with V
        self.query_layer = nn.Linear(msg_dim, att_msg_dim)
        self.key_layer   = nn.Linear(msg_dim, att_msg_dim)
        self.value_layer = nn.Linear(msg_dim, att_msg_dim)
 
        self.out_proj = nn.Linear(att_msg_dim, att_msg_dim)
        self.norm = nn.LayerNorm(att_msg_dim)
 
    def forward(self, obs_batch):
        obs_batch = obs_batch.to(next(self.parameters()).device)  
        B, N, _ = obs_batch.size()
 
        # Linear projections
        msg_vecs = self.msg_encoder(obs_batch)
        Q = self.query_layer(msg_vecs)  
        K = self.key_layer(msg_vecs)
        V = self.value_layer(msg_vecs)
 
        # Split thành multi-head: [B, N, num_heads, head_dim]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
 
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
 
        msg_agg = torch.matmul(attn_weights, V)
 
        msg_agg = msg_agg.transpose(1, 2).contiguous().view(B, N, self.att_msg_dim)
 
        res = msg_agg
        msg_agg = self.out_proj(msg_agg)
        msg_agg = self.norm(msg_agg + res)
 
        return torch.cat([msg_vecs, msg_agg], dim=-1)
 
class ReplayBufferGRU:
    """
    Replay buffer for agent with GRU network
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
   
    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, greedy_action):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, state, action, last_action, reward, next_state, greedy_action
        )
        self.position = int((self.position +1) % self.capacity)
   
    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, greedy_lst = [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, greedy_action = sample
            #h_in = [1, batchsize= 1 , n_agents, hidden_size]
            min_seq_len = min(len(state), min_seq_len)
            hi_lst.append(h_in)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()
        ho_lst = torch.cat(ho_lst, dim=-3).detach()
 
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, greedy_action = sample
            sample_len = len(state)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx + min_seq_len
            s_lst.append(state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            la_lst.append(last_action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
            greedy_lst.append(greedy_action[start_idx:end_idx])
       
        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst, greedy_lst
   
    def ___len__(self):
        return len(self.buffer)
   
    def get_length(self):
        return len(self.buffer)
           
 
class RNNAgent(nn.Module):
    '''
    @  This class evaluate the Q value given a state + action
    '''
    def  __init__(self, num_inputs, action_shape, num_actions, hidden_size, msg_dim, hidden_dim_for_att, att_msg_dim):
        super(RNNAgent, self).__init__()
        self.num_inputs = num_inputs
        self.action_shape = action_shape
        self.num_actions = num_actions
        self.msg_dim = msg_dim
        self.att_msg_dim = att_msg_dim
        self.hidden_dim_for_att = hidden_dim_for_att
        self.hidden_size = hidden_size
        # print("W.shape = ", num_inputs+action_shape*num_actions)
        self.linear1 = nn.Linear(num_inputs+action_shape*num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)  
        self.attn_comm = AttentionComm_v2(obs_dim=hidden_size, hidden_dim_1=hidden_dim_for_att, msg_dim=msg_dim, att_msg_dim=att_msg_dim)
 
    def forward(self, state, action, hidden_in, debug=False):
        '''
        @params:
            state: [#B, #S, #Agent, n_feature]
            action: [#B, #S, #Agent, action_shape]
        @return:
            x: [#S, #B, #Agent, #(msg_dim + att_msg_dim)]
            hidden_out: [#1, #B, #Agent, hidden_dim]
        '''      
        # print("state.shape  = ", state.shape)
        if debug:
            print("DEBUG forward:")
            print(" state.device =", state.device, " state.dtype =", state.dtype, " state.shape =", state.shape)
            print(" action.device =", action.device, " action.dtype =", action.dtype, " action.shape =", action.shape)
 
        # print("action = ", action.shape)
        bs, seq_len, n_agents, _ = state.shape
        state = state.permute(1,0,2,3) # state = [#S, #B, #Agent, n_feature]
        action = action.permute(1,0,2,3) # action = [#S, #B, #Agent, n_feature]
        # print("action.shape in forward", action.shape)
        # print("action before one_hot:", action, action.dtype, action.shape)
        action = F.one_hot(action, num_classes=self.num_actions) # action = [#S, #B, #Agent, n_feature, num_actions]
        action = action.view(seq_len, bs, n_agents, -1)
        # -> action = [#S, #B, #Agent, n_feature * num_actions]
        x = torch.cat([state, action.float()], -1)
        # -> x = [#S, #B, #Agent, n_feature * num_actions + n_feature]
        x = x.view(seq_len, bs*n_agents, -1)
        hidden_in = hidden_in.view(1, bs*n_agents, -1)

        # print("x.shape = ", x.shape)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # -> x = [#S, #B, #Agent, hidden_size]
        # print("x.shape = ", x.shape)
        x, hidden = self.rnn(x, hidden_in)

        #attention
        x = x.view(seq_len * bs, n_agents, self.hidden_size)
        x = self.attn_comm(x)

        x = x.view(seq_len, bs, n_agents, self.msg_dim + self.att_msg_dim)
        return x, hidden

class QLocal(nn.Module):
    '''
    This model receives the commnunicated infomation from agents and return the local Q values.
    '''
    def __init__(self, state_dim, hidden_size, action_shape, num_actions):
        super().__init__()
        self.state_dim = state_dim
        self.layer = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.action_shape = action_shape
        self.num_actions = num_actions
    def forward(self, state):
        """
        @params:
            state: [#seq, #batch, #n_agents, #state_dim]
        @return:
            qs: [#B, #S, #Agent, action_shape, num_actions]
        """
        # modify 
        seq_len, bs, n_agents, _ = state.shape
        x = self.layer(state)
        # [#S, #B, #agent, action_shape * num_actions]
        x = x.view(seq_len, bs, n_agents, self.action_shape, self.num_actions)
        qs = x.permute(1,0,2,3,4)  # [B, S, Agent, action_shape, num_actions]
        return qs
 
class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, action_shape, embed_dim=128, hypernet_embed=256, abs=True):
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
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)  # [#batch*#sequence, self.embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)  # [#batch*#sequence, self.embed_dim, 1]
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # [#batch*#sequence, 1, 1]
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # [#batch, #sequence, 1]
        # q_tot = (-750)*(torch.tanh(q_tot) + 1)
        # q_tot = torch.tanh(q_tot) * 2000
        return q_tot

class QMix_Trainer(nn.Module):
    def __init__(self, replay_buffer,
                 n_agents, state_dim, action_shape, action_dim, hidden_dim, msg_dim, att_msg_dim, hidden_dim_for_att,
                 hypernet_dim, target_update_interval,
                 lr=0.001, grad_clip_norm=10.0, max_steps = 1e4, logger=None):
        super(QMix_Trainer, self).__init__()
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.max_steps = 1e4
        self.target_update_interal = target_update_interval
        # def  __init__(self, num_inputs, action_shape, num_actions, hidden_size, msg_dim, hidden_dim_for_att, att_msg_dim):
        self.rnn = RNNAgent(num_inputs=state_dim,
                        action_shape=action_shape,
                        num_actions=action_dim,
                        hidden_size=hidden_dim,
                        msg_dim=msg_dim,
                        hidden_dim_for_att=hidden_dim_for_att,
                        att_msg_dim=att_msg_dim).to(device)
        self.target_rnn = RNNAgent(num_inputs=state_dim,
                        action_shape=action_shape,
                        num_actions=action_dim,
                        hidden_size=hidden_dim,
                        msg_dim=msg_dim,
                        hidden_dim_for_att=hidden_dim_for_att,
                        att_msg_dim=att_msg_dim).to(device)
        # def __init__(self, state_dim, hidden_size):
        self.q_local = QLocal(state_dim=msg_dim + att_msg_dim,
                                hidden_size=hidden_dim, action_shape=action_shape, num_actions=action_dim).to(device)
        self.target_q_local = QLocal(state_dim=msg_dim + att_msg_dim,
                                hidden_size=hidden_dim, action_shape=action_shape, num_actions=action_dim).to(device)
        # def __init__(self, state_dim, n_agents, action_shape, embed_dim=128, hypernet_embed=256, abs=True):
        self.mixer = QMix(state_dim=msg_dim + att_msg_dim,
                            n_agents=n_agents,
                            action_shape=action_shape,
                            embed_dim=hidden_dim,
                            hypernet_embed=hypernet_dim).to(device)
        self.target_mixer = QMix(state_dim=msg_dim + att_msg_dim,
                            n_agents=n_agents,
                            action_shape=action_shape,
                            embed_dim=hidden_dim,
                            hypernet_embed=hypernet_dim).to(device)
        self.scaler = GradScaler()   
 
 
        self._update_targets()
        self.update_cnt = 0
        self.criterion = nn.SmoothL1Loss()
 
        self.optimizer = optim.Adam(
            list(self.rnn.parameters()) + list(self.q_local.parameters()) + list(self.mixer.parameters()), lr=lr
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.grad_clip_norm = grad_clip_norm
        self._lambda = 0
        print(f"Tổng số tham số trainable: {total_params}")

    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.n_agents, self.action_shape))

        return action.type(torch.FloatTensor).numpy()


    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @return:
            action: w/ shape [#active_as]
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device) # add #sequence and #batch: [[#batch, #sequence, n_agents, n_feature]] 
        last_action = torch.LongTensor(
            last_action).unsqueeze(0).unsqueeze(0).to(device)  # add #sequence and #batch: [#batch, #sequence, n_agents, action_shape]
 
        hidden_in = hidden_in.unsqueeze(1)
        x, hidden_out = self.rnn(state, last_action, hidden_in)  # [B,S,Agent,A,K]
        logits = self.q_local(x)
        probs = F.softmax(logits, dim=-1)   # convert to probs for sampling
        dist = Categorical(probs)
        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze(0).squeeze(0).detach().cpu().numpy()
        return action, hidden_out

        return action, hidden_out

    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_state, episode_action,
                            episode_last_action, episode_reward, episode_next_state, episode_greedy_action):
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action,
                                episode_last_action, episode_reward, episode_next_state, episode_greedy_action)
    
    def update(self, batch_size, mode):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, greedy_action = self.replay_buffer.sample(batch_size)
        state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device, non_blocking=True)
        next_state = torch.from_numpy(np.array(next_state, dtype=np.float32)).to(device, non_blocking=True)
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(-1).to(device, non_blocking=True)
        hidden_in = hidden_in.to(device)
        hidden_out = hidden_out.to(device)
        action = torch.from_numpy(np.array(action)).long().to(device, non_blocking=True)
        last_action = torch.from_numpy(np.array(last_action)).long().to(device, non_blocking=True)
        greedy_action = torch.from_numpy(np.array(greedy_action)).long().to(device, non_blocking=True)
 
        # ======= AMP AUTCAST =======
        with autocast(dtype=torch.bfloat16):  
            # Forward agent
            x, _ = self.rnn(state, last_action, hidden_in, debug=False)
            # x : [S, B, N, hidden_dim]
            agent_outs = self.q_local(x)
            x.permute(1,0,2,3)
            # x : [B, S, N, hidden_dim]
            chosen_action_qvals = torch.gather(agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
            qtot = self.mixer(chosen_action_qvals, x)
 
            # Target network, no_grad
            with torch.no_grad():
                target_x, _ = self.target_rnn(next_state, action, hidden_out)
                target_agent_outs = self.target_q_local(target_x)
                target_x.permute(1,0,2,3)
                # target_x : [B, S, N, hidden_dim]                
                target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0]
                target_qtot = self.target_mixer(target_max_qvals, target_x)
 
            reward = reward[:,:,0]
            targets = self._build_td_lambda_targets(reward, target_qtot)
            if mode == 'self-learning':
                td_loss_raw = self.criterion(qtot, targets.detach())
                loss = td_loss_raw
            else:
                log_probs = F.log_softmax(agent_outs, dim=-1) 
                distill_loss_raw = F.nll_loss(
                    log_probs.reshape(-1, log_probs.size(-1)),   # [B*S*N*A, K]
                    greedy_action.reshape(-1),                   # [B*S*N*A]
                    reduction='mean'
                )
                loss = distill_loss_raw
 
        # Backward với GradScaler
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Unscale trước khi clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
 
        self.update_cnt +=1
        if self.update_cnt % self.target_update_interal == 0:
            self._update_targets()
 
        return loss.item()
 
    def _build_td_lambda_targets(self, rewards, target_qs, gamma=0.99, td_lambda=0.6):
        rewards = rewards.unsqueeze(-1)
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1]
 
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1])
 
        return ret
   
    def _update_targets(self):
        #Update target networks
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(param.data)
       
        for target_param, param in zip(self.target_q_local.parameters(), self.q_local.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_rnn.parameters(), self.rnn.parameters()):
            target_param.data.copy_(param.data)
 
    def save_model(self, path):
        torch.save(self.q_local.state_dict(), path+'_qlocal')
        torch.save(self.mixer.state_dict(), path+'_mixer')
        torch.save(self.rnn.state_dict(), path+'_rnn')
 
    def load_model(self, path):
        self.q_local.load_state_dict(torch.load(path+'_qlocal', map_location="cuda"))
        self.mixer.load_state_dict(torch.load(path+'_mixer', map_location="cuda"))
        self.rnn.load_state_dict(torch.load(path+'_rnn', map_location="cuda"))
