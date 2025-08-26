from model import QMix_Trainer 
from coor_env import CoorEnv
import random
import numpy as np
import torch
from model import ReplayBufferGRU
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Train or test QMIX')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    env = CoorEnv(num_steps=1, seed=0)
    obs = env.base_env.reset()
    n_agents, obs_dim = obs.shape[0], obs.shape[1]
    action_shape = 1
    action_dim = env.action_dim
    replay_buffer_size = int(1e6)
    hidden_dim = 256 
    att_msg_dim = 1024
    hypernet_dim = 512
    max_episodes = 500
    update_iter  = 32
    batch_size = 96
    save_interval = 50
    msg_dim = 512
    hidden_dim_for_att = 512
    target_update_interval = 300
    grad_clip_norm = 10.0 
    max_steps = 10000
    learning_rate = 1e-4
    model_path = 'non_teacher/qmix'
    model_load = 'model/qmix'
    writer = SummaryWriter(log_dir="runs/qmix_experiment")
    replay_buffer = ReplayBufferGRU(replay_buffer_size)
    trainer = QMix_Trainer(
        replay_buffer, n_agents, obs_dim, action_shape, action_dim,
        hidden_dim, msg_dim, att_msg_dim, hidden_dim_for_att, hypernet_dim, target_update_interval,
        lr=learning_rate, grad_clip_norm=grad_clip_norm, max_steps=max_steps
    )
    # trainer.load_model(model_load)
    print(f"load model done from {model_load}")
    count_change = 0

    for epi in range(max_episodes):
        loss = 0.0
        loss_total = 0.0
        hidden_out = torch.zeros([1, n_agents, hidden_dim], dtype=torch.float).to(device)
        last_action = trainer.sample_action()

        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_greedy_action = []

        state = env.reset()
        number_of_steps = 0

        for step in range(max_steps):
            # print("step = ", step)
            hidden_in = hidden_out
            action, hidden_out = trainer.get_action(
                state, last_action, hidden_in
            )
            greedy_action = env._get_greedy_action()
            greedy_action = np.array(greedy_action).reshape(-1, 1)
            next_state, reward, done, info, used = env.step(action.reshape(-1))
            number_of_steps += 1
            if step==0:
                ini_hidden_in =  hidden_in
                ini_hidden_out = hidden_out
            episode_state.append(state)
            episode_action.append(action.astype(np.int64))
            episode_last_action.append(last_action.astype(np.int64))
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_greedy_action.append(greedy_action)
            
            state = next_state

            last_action = action

            if np.any(done):
                break
        
        if args.train:
            trainer.push_replay_buffer(ini_hidden_in, ini_hidden_out,
                                       episode_state, episode_action, episode_last_action,
                                       episode_reward, episode_next_state, episode_greedy_action)
            if epi > batch_size + 32:
                count_change += 1
            
                for _ in range(update_iter):
                    loss = trainer.update(batch_size, mode='teacher')
                    loss_total += loss

                trainer._lambda = min(float(count_change/100), 1.0)
            if epi % save_interval == 0:
                trainer.save_model(model_path)
        if loss_total:
            print(f"Episode: {epi}, Episode Reward: {np.mean(episode_reward) + 1.0}, Total Loss: {loss_total/update_iter}, Lambda: {trainer._lambda}")
        else:
            print(f"Episode: {epi}, Episode Reward: {np.mean(episode_reward) + 1.0}, Total Loss: {loss_total}, Lambda: {trainer._lambda}")
        
        writer.add_scalar("Reward/Episode", np.mean(episode_reward) + 1.0, epi)
        writer.add_scalar("Loss/Episode", loss_total/update_iter, epi)

    writer.close()


if __name__ == "__main__":
    main()
