from model import QMix_Trainer 
from coor_env import CoorEnv
import random
import numpy as np
import torch
from model import ReplayBufferGRU
import time
import argparse

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    env = CoorEnv(num_steps=10, seed=0)
    obs = env.base_env.reset()
    n_agents, obs_dim = obs.shape[0], obs.shape[1]
    action_shape = 1
    action_dim = env.action_dim


    replay_buffer_size = 1e4
    hidden_dim = 64
    hypernet_dim = 128
    max_steps = 1000
    max_episodes = 1000
    update_iter  = 1
    batch_size = 2
    save_interval = 10
    msg_dim = 64
    target_update_interval = 10
    model_path = 'model/qmix'
    replay_buffer = ReplayBufferGRU(replay_buffer_size)

    trainer = QMix_Trainer(
        replay_buffer, n_agents, obs_dim, action_shape, action_dim, hidden_dim, msg_dim, hypernet_dim, target_update_interval
    )

    loss = None

    for epi in range(max_episodes):
        hidden_out = torch.zeros([1, n_agents, hidden_dim], dtype=torch.float).to(device)
        last_action = trainer.sample_action()
        # print("last_action = ", last_action)

        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []

        state = env.reset()

        for step in range(max_steps):
            print("step = ", step)
            hidden_in = hidden_out
            action, hidden_out = trainer.get_action(
                state, last_action, hidden_in
            )
            next_state, reward, done, info, used = env.step(action.reshape(-1))
            # print(f"Commander step {step}: reward={reward:.4f}, used={used} base steps, done={done}")

            if step==0:
                ini_hidden_in =  hidden_in
                ini_hidden_out = hidden_out
            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward / 1000.0)
            episode_next_state.append(next_state)
            
            state = next_state
            last_action = action

            if np.any(done):
                break
        
        if args.train:
            trainer.push_replay_buffer(ini_hidden_in, ini_hidden_out,
                                       episode_state, episode_action, episode_last_action,
                                       episode_reward, episode_next_state)
            if epi > batch_size:
                print("--------START TRAINING ---------------")
                for _ in range(update_iter):
                    loss = trainer.update(batch_size)
            
            if epi % save_interval == 0:
                trainer.save_model(model_path)
        
        print(f"Episode: {epi}, Episode Reward: {np.sum(episode_reward)}, Loss: {loss}")


if __name__ == "__main__":
    main()