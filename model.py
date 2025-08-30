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
class WorldModel(nn.Module):
    '''
    WorldModel: predictions for the states of targets
      - Self-Attention over agents
      - Cross-Attention from agents to targets 
      - GRU for temporal dynamics
    '''
    def __init__(self, n_agents, n_targets, local_state_dim, target_state_dim, hidden_size, num_heads=4):
        super(WorldModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.agent_encoder = nn.Sequential(
                                nn.Linear(local_state_dim, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size))
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.linear_middle = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size))

        # Embedding learnable 
        self.target_emb = nn.Parameter(torch.randn(n_targets, hidden_size))

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        # Output projection
        self.out_proj = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, target_state_dim))
                            

    def forward(self, states, hidden):
        """
        agents: [B, Seq, n_agents, local_state_dim]
        hidden: hidden state cho GRU [1, B, H] 
        return: 
          pred: [B, Seq, n_targets, target_state_dim]
          hidden: GRU hidden state
        """
        bs, seq, n_agents, _ = states.shape
        states = states.permute(1,0,2,3)
        # states: [Seq, Batch, n_agents, local_state_dim]
        states = states.view(seq * bs, n_agents, -1)
        # states: [Seq * bs, n_agents, local_state_dim]
        agent_emb = self.agent_encoder(states)  
        # agent_emb: [Seq * bs, n_agents, hidden_size]
        agent_emb, _ = self.self_attn(agent_emb, agent_emb, agent_emb) 
        # agent_emb: [Seq * Batch, n_agents, hidden_size]
        agent_emb = agent_emb.view(seq, bs * n_agents, -1)
        # agent_emb : [Seq, bs * n_agents, hidden_size] for GRU
        out, hidden_out = self.rnn(agent_emb, hidden)  
        # out: [Seq, bs*n_agents, hidden_size]
        out = out.view(seq*bs, n_agents, -1)
        # out: [seq * bs, n_agents, hidden_size]
        out = self.linear_middle(out)
        target_emb = self.target_emb.unsqueeze(0).repeat(bs * seq, 1, 1)  # [B*s, n_targets, H]

        out, _ = self.cross_attn(query=target_emb, key=out, value=out)  # [B*s, n_targets, H]

        pred = self.out_proj(out)  # [B*s, n_targets, target_state_dim]

        pred = pred.view(bs, seq, self.n_targets, -1)

        return pred, hidden_out
    
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
    Replay buffer for agent with GRU networks
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
   
    def push(self, hidden_in, hidden_out, hidden_in_wm, hidden_out_wm, state, action, last_action, reward, next_state, greedy_action, target_states):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, hidden_in_wm, hidden_out_wm, state, action, last_action, reward, next_state, greedy_action, target_states
        )
        self.position = int((self.position +1) % self.capacity)
   
    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, hi_wm_lst, ho_wm_lst, greedy_lst, target_lst = [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, h_in_wm, h_out_wm, state, action, last_action, reward, next_state, greedy_action, target_states = sample
            #h_in = [1, batchsize= 1 , n_agents, hidden_size]
            min_seq_len = min(len(state), min_seq_len)
            hi_lst.append(h_in)
            ho_lst.append(h_out)
            hi_wm_lst.append(h_in_wm)
            ho_wm_lst.append(h_out_wm)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()
        ho_lst = torch.cat(ho_lst, dim=-3).detach()
        hi_wm_lst = torch.cat(hi_lst, dim=-3).detach()
        ho_wm_lst = torch.cat(ho_lst, dim=-3).detach()
 
        for sample in batch:
            h_in, h_out, h_in_wm, h_out_wm, state, action, last_action, reward, next_state, greedy_action = sample
            sample_len = len(state)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx + min_seq_len
            s_lst.append(state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            la_lst.append(last_action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
            greedy_lst.append(greedy_action[start_idx:end_idx])
            target_lst.append(target_states[start_idx:end_idx])
       
        return hi_lst, ho_lst, hi_wm_lst, ho_wm_lst, s_lst, a_lst, la_lst, r_lst, ns_lst, greedy_lst, target_lst
   
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

class DMAQ_SI_Weight(nn.Module):
    """
    This function takes the lambdas from state and action
    @params:
        state: [Batch, Seq, n_agents, state_dim]
        action: [Batch, Seq, n_agents, action_dim]
    @return:
        lambda: [Batch, Seq, n_agents]
    """
    def __init__(self, n_agents, n_actions, local_state_dim, num_kernel, adv_hypernet_embed):
        super(DMAQ_SI_Weight, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.local_state_dim = local_state_dim
        self.state_dim = n_agents * self.local_state_dim
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim 

        self.num_kernel = num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        self.adv_hypernet_embded = adv_hypernet_embed

        for _ in range(self.num_kernel): #multi-head attention
            self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, self.adv_hypernet_embded),
                                                     nn.ReLU(),
                                                     nn.Linear(self.adv_hypernet_embded, 1))) #key
            
            self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, self.adv_hypernet_embded),
                                                     nn.ReLU(),
                                                     nn.Linear(self.adv_hypernet_embded, self.n_agents))) #agent
            self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, self.adv_hypernet_embded),
                                                     nn.ReLU(),
                                                     nn.Linear(self.adv_hypernet_embded, self.n_agents))) #action            
            

    def forward(self, states, actions):
        """
        @params:
            actions: [#B, #S, #Agent, action_shape]
        """
        #states: [B, S, n_agents, local_state_dim]
        bs, seq_len, n_agents, _ = states.shape
        states = states.reshape(-1, self.state_dim) # self.state_dim = n_agents * local_state_dim
        #states: [B * S, self.state_dim]

        actions = F.one_hot(actions, num_classes=self.n_actions)
        actions = actions.view(bs, seq_len, n_agents, -1)
        #actions: [B, S, n_agents, action_shape * self.n_actions]
        #since the action_shape = 1 -> actions = [B, S, n_agents, self.n_actions]
        #since the self.action_dim = n_agents * self.n_actions
        actions = actions.view(-1, self.action_dim)
        #actions : [B * S, self.action_dim]
        data = torch.cat([states, actions], dim=-1)
        #data : [B * S, self.state_action_dim] 

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        all_head_action = [self_ext(data) for self_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = torch.abs(curr_head_key).repeat(1, self.n_agents)  + 1e-10
            # x_key : [B * S, self.n_agents]
            x_agents = torch.sigmoid(curr_head_agents)
            # x_agents : [B * S, self.n_agents]
            x_action = torch.sigmoid(curr_head_action)
            # x_action : [B * S, self.n_agents]
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)
        
        head_attend = torch.stack(head_attend_weights, dim=1)
        # head_attend : [B * S, num_kernels, self.n_agents]
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = torch.sum(head_attend, dim=1)
        # lấy sum theo chiều num_kernel
        # head_attend = [B * S, self.n_agents]
        return head_attend
    
class DMAQer(nn.Module):
    '''
    Dueling Mixing Net
        Transformation Net
        Dueling Mixing Net
    '''
    def __init__(self, n_agents, n_actions, local_state_dim, num_kernel, mixing_embed_dim, hypernet_embed, adv_hypernet_embed):
        super(DMAQer, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.local_state_dim = local_state_dim
        self.state_dim = n_agents * self.local_state_dim
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.embed_dim = mixing_embed_dim
        self.hypernet_embed = hypernet_embed
        self.num_kernel = num_kernel
        self.adv_hypernet_embed = adv_hypernet_embed

        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(self.hypernet_embed, self.n_agents))
        
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(self.hypernet_embed, self.n_agents))
        
        self.si_weight = DMAQ_SI_Weight(n_agents=self.n_agents,
                                        n_actions=self.n_actions,
                                        local_state_dim=self.local_state_dim,
                                        num_kernel=self.num_kernel,
                                        adv_hypernet_embed= self.adv_hypernet_embed)
    
    def calc_v(self, agent_qs):
        #This function calculate V_tot form agent_qs 
        # via the equation (not true, just for easy to code :> )
        # V = total qs
        #agent_qs : [B, S, n_agents]
        agent_qs = agent_qs.view(-1, self.n_agents)
        #agent_qs : [B *S, n_agents]
        v_tot = torch.sum(agent_qs, dim=-1)
        #v_tot = [B * S]
        return v_tot
    
    def calc_adv(self, agent_qs, states, actions, max_q_i):
        # states = states.view(-1, self.state_dim)
        # actions = actions.view(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)
        
        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
        #adv_tot: [B* S]

        return adv_tot
    
    def calc(self, agent_qs, states, actions, max_q_i, is_v):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        # print("debug : agent_qs.shape = ", agent_qs.shape)
        bs, seq, _ = agent_qs.shape
        # print("debug : states.shape = ", states.shape)
        bs, seq, nu_agents, _ = states.shape
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w_final = self.hyper_w_final(states)
        w_final = torch.abs(w_final)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)
        v = v.view(-1, self.n_agents)

        agent_qs = w_final * agent_qs + v 
        #calculate v_i
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v
        
        states = states.view(bs, seq, nu_agents, -1)
        result = self.calc(agent_qs=agent_qs, states=states, actions=actions, max_q_i=max_q_i, is_v=is_v)

        result = result.view(bs, seq, 1)

        return result


class QMix_Trainer(nn.Module):
    def __init__(self, replay_buffer,
                 n_agents, n_targets, state_dim, target_state_dim, action_shape, action_dim, hidden_dim, msg_dim, att_msg_dim, hidden_dim_for_att,
                 hypernet_dim, target_update_interval, num_kernel, 
                 lr=0.001, grad_clip_norm=10.0, max_steps = 1e4, logger=None):
        super(QMix_Trainer, self).__init__()
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.action_shape = action_shape
        self.num_kernel = num_kernel
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.max_steps = 1e4
        self.target_update_interal = target_update_interval
        self.adv_hypernet_embed = hidden_dim
        self.target_state_dim = target_state_dim
        # def __init__(self, n_agents, n_targets, local_state_dim, target_state_dim, hidden_size, num_heads=4):
        self.world_model = WorldModel(n_agents=n_agents,
                                      n_targets=n_targets,
                                      local_state_dim=state_dim,
                                      target_state_dim=target_state_dim,
                                      hidden_size=hidden_dim,
                                      num_heads=4)
        # def  __init__(self, num_inputs, action_shape, num_actions, hidden_size, msg_dim, hidden_dim_for_att, att_msg_dim):
        self.rnn = RNNAgent(num_inputs=state_dim + n_targets * target_state_dim,
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
        # def __init__(self, n_agents, n_actions, local_state_dim, num_kernel, mixing_embed_dim, hypernet_embed):
        self.mixer = DMAQer(n_agents=n_agents, 
                            n_actions=action_dim, 
                            local_state_dim=msg_dim + att_msg_dim,
                            num_kernel=num_kernel,
                            mixing_embed_dim=hidden_dim,
                            hypernet_embed=hypernet_dim,
                            adv_hypernet_embed=hidden_dim
                            ).to(device)
        self.target_mixer = DMAQer(n_agents=n_agents, 
                            n_actions=action_dim, 
                            local_state_dim=msg_dim + att_msg_dim,
                            num_kernel=num_kernel,
                            mixing_embed_dim=hidden_dim,
                            hypernet_embed=hypernet_dim,
                            adv_hypernet_embed=hidden_dim
                            ).to(device)
        self.scaler = GradScaler()   
 
        self._update_targets()
        self.update_cnt = 0
        self.criterion = nn.MSELoss()
 
        self.optimizer = optim.Adam(
            list(self.world_model.parameters()) + list(self.rnn.parameters()) + list(self.q_local.parameters()) + list(self.mixer.parameters()), lr=lr
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.grad_clip_norm = grad_clip_norm
        print("grad_norm = ", self.grad_clip_norm)
        self._lambda = 0
        print(f"Tổng số tham số trainable: {total_params}")

    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.n_agents, self.action_shape))

        return action.type(torch.FloatTensor).numpy()
    
    def get_action(self, state, last_action, hidden_in, hidden_in_wm, deterministic=False):
        '''
        @return:
            action: w/ shape [#active_as]
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device) # add #sequence and #batch: [[#batch, #sequence, n_agents, n_feature]] 
        last_action = torch.LongTensor(
            last_action).unsqueeze(0).unsqueeze(0).to(device)  # add #sequence and #batch: [#batch, #sequence, n_agents, action_shape]
 
        hidden_in = hidden_in.unsqueeze(1)
        hidden_in_wm = hidden_in_wm.unsqueeze(1)
        predictions, hidden_out_wm = self.world_model(state, hidden_in_wm)

        #prediction processing
        bs, seq, n_targets, _ = predictions.shape
        predictions = predictions.view(bs, seq, n_targets * self.target_state_dim)
        predictions = predictions.unsqueeze(2)
        predictions = predictions.repeat(1, 1, self.n_agents, 1)
        ####
        state = torch.cat([state, predictions], dim=-1)
        #state : [B , S , n_agents, state_dim + n_targets * self.targets_state_dim]
        x, hidden_out = self.rnn(state, last_action, hidden_in)  # [B,S,Agent,A,K]
        logits = self.q_local(x)
        probs = F.softmax(logits, dim=-1)   # convert to probs for sampling
        dist = Categorical(probs)
        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze(0).squeeze(0).detach().cpu().numpy()
        return action, hidden_out, hidden_out_wm
    
    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, ini_hidden_in_wm, ini_hidden_out_wm, episode_state, episode_action,
                            episode_last_action, episode_reward, episode_next_state, episode_greedy_action, episode_target_states):
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, ini_hidden_in_wm, ini_hidden_out_wm, episode_state, episode_action,
                                episode_last_action, episode_reward, episode_next_state, episode_greedy_action, episode_target_states)
        
    def update(self, batch_size, mode):
        hidden_in, hidden_out, hidden_in_wm, hidden_out_wm, state, action, last_action, reward, next_state, greedy_action, target_states = self.replay_buffer.sample(batch_size)
        state = torch.from_numpy(np.array(state, dtype=np.float32)).to(device, non_blocking=True)
        next_state = torch.from_numpy(np.array(next_state, dtype=np.float32)).to(device, non_blocking=True)
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(-1).to(device, non_blocking=True)
        hidden_in = hidden_in.to(device)
        hidden_out = hidden_out.to(device)
        hidden_in_wm = hidden_in_wm.to(device)
        hidden_out_wm = hidden_out_wm.to(device)
        action = torch.from_numpy(np.array(action)).long().to(device, non_blocking=True)
        last_action = torch.from_numpy(np.array(last_action)).long().to(device, non_blocking=True)
        greedy_action = torch.from_numpy(np.array(greedy_action)).long().to(device, non_blocking=True)
        target_states = torch.from_numpy(np.array(target_states, dtype=np.float32)).to(device, non_blocking=True)

        # ======= AMP AUTCAST =======
        with autocast():  
            predictions, _ = self.world_model(state, hidden_in_wm)
            bs, seq, n_targets, _ = predictions.shape
            predictions = predictions.view(bs, seq, n_targets * self.target_state_dim)

            merged_predictions = predictions.unsqueeze(2)
            merged_predictions = merged_predictions.repeat(1, 1, self.n_agents, 1)

            state = torch.cat([state, merged_predictions.detach()], dim=-1)
            x, _ = self.rnn(state, last_action, hidden_in, debug=False)
            agent_outs = self.q_local(x)
            x = x.permute(1,0,2,3)
            chosen_action_qvals = torch.gather(agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
            #chosen_action_qvals : [B, S, N, action_shape]
            # since the action_shape = 1
            chosen_action_qvals = chosen_action_qvals.squeeze(-1)
            # chosen_action_qvals : [B, S, N]
            v_tot = self.mixer(chosen_action_qvals, x, is_v=True)
            #agent_outs = [B, S, N, action_shape, num_action]
            agent_outs_detached = agent_outs.clone().detach()
            agent_outs_detached = agent_outs_detached.squeeze(-2)
            max_action_qvals, max_action_index = agent_outs_detached.max(dim=3)
            a_tot = self.mixer(chosen_action_qvals, x, actions=last_action, max_q_i=max_action_qvals, is_v=False)
            qtot = v_tot + a_tot

            #x: [B, S, N, -1]
            target_x = x.clone().detach()
 
            # Target network, no_grad
            with torch.no_grad():
                target_x = target_x.permute(1,0,2,3)
                #target_x : [S, B, N, -1]
                target_agent_outs = self.target_q_local(target_x)
                target_x = target_x.permute(1,0,2,3)   
                #target_x : [B, S, N, -1]          
                target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0]
                target_max_qvals = target_max_qvals.squeeze(-1).squeeze(-1)
                target_v_tot = self.target_mixer(target_max_qvals, target_x, is_v=True)
                target_a_tot = self.target_mixer(target_max_qvals, target_x, action, max_q_i=target_max_qvals, is_v=False)
                target_qtot = target_v_tot + target_a_tot


            reward = reward[:,:,0]
            targets = self._build_td_lambda_targets(reward, target_qtot)
            pred_loss = F.mse_loss(predictions, target_states.detach())
            if mode == 'self-learning':
                td_loss_raw = self.criterion(qtot, targets.detach())
                loss = td_loss_raw
            elif mode == 'teacher':
                log_probs = F.log_softmax(agent_outs, dim=-1) 
                distill_loss_raw = F.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)),   # [B*S*N*A, K]
                    greedy_action.view(-1),                   # [B*S*N*A]
                    reduction='mean'
                )
                loss = distill_loss_raw
            elif mode == 'both':
                td_loss_raw = self.criterion(qtot, targets.detach())
                log_probs = F.log_softmax(agent_outs, dim=-1) 
                distill_loss_raw = F.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)),   # [B*S*N*A, K]
                    greedy_action.view(-1),                   # [B*S*N*A]
                    reduction='mean'
                )
                loss = 0.9 * td_loss_raw + 0.1 * distill_loss_raw + pred_loss
 
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
 
        return loss.item(), pred_loss
 
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
 
    def save_model(self, path):
        torch.save(self.q_local.state_dict(), path+'_qlocal')
        torch.save(self.mixer.state_dict(), path+'_mixer')
        torch.save(self.rnn.state_dict(), path+'_rnn')
        torch.save(self.world_model.state_dict(), path+'_wm')

 
    def load_model(self, path):
        self.q_local.load_state_dict(torch.load(path+'_qlocal', map_location="cuda"))
        self.mixer.load_state_dict(torch.load(path+'_mixer', map_location="cuda"))
        self.rnn.load_state_dict(torch.load(path+'_rnn', map_location="cuda"))
        self.world_model.load_state_dict(torch.load(path+'_wm', map_location="cuda"))
