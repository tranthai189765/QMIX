import mate
import gym
import numpy as np
import torch
import math
import random

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

def compute_alpha(agent_pos, agent_dir, target_pos):
    dx, dy = target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1]
    angle_to_target = math.atan2(dy, dx)
    alpha = normalize_angle(angle_to_target - agent_dir)
    return alpha

def compute_theta(agent_pos, agent_view, seen_targets):
    """
    calculate theta: the smallest angle for agent to cover all seen_targets
    return a delta (how much we need to change viewing angle width)
    """
    if not seen_targets:
        return 0.0
    angles = [math.atan2(t[1] - agent_pos[1], t[0] - agent_pos[0]) for t in seen_targets]
    angles.sort()
    diffs = []
    n = len(angles)
    for i in range(n):
        j = (i + 1) % n
        diff = normalize_angle(angles[j] - angles[i])
        if diff < 0:
            diff += 2 * math.pi
        diffs.append(diff)
    max_gap = max(diffs)
    theta_needed = 2 * math.pi - max_gap
    return agent_view - theta_needed

class CoorEnv(gym.Env):
    """
    Môi trường phối hợp phân công mục tiêu (discrete bitmask hành động)
    bọc quanh base_env của MATE (liên tục delta_theta, delta_alpha).
    """
    metadata = {"render.modes": []}
    from typing import Optional

    def __init__(self, num_steps=10, seed: Optional[int] = None):
        super().__init__()
        base_env = mate.make('MultiAgentTracking-v0')
        base_env = mate.MultiCamera(base_env, target_agent=mate.GreedyTargetAgent(seed=0))
        self.base_env = base_env

        self.num_agents = base_env.num_teammates
        self.num_targets = base_env.num_opponents
        self.action_dim = 2 ** self.num_targets  # bitmask phân công targets cho từng agent
        self.action_space = gym.spaces.MultiDiscrete([self.action_dim] * self.num_agents)

        # Quan sát giữ nguyên theo base_env
        self.observation_space = self.base_env.observation_space

        self.num_steps = int(num_steps)
        self._t = 0
        self._last_obs = None
        if seed is not None:
            self.seed(seed)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed or 0)
        try:
            self.base_env.seed(seed)
        except Exception:
            pass
        return [seed]

    def reset(self):
        self._t = 0
        obs = self.base_env.reset()
        self._last_obs = obs
        return obs

    def _extract_position(self, state):
        """
        extract positions from state
        @return:
            agent_positions + current view: [num_agents, 4]
            target_positions: [{'seen': bool, 'pos': (x, y) or None}, ...]
        """
        agent_positions = []
        target_positions = [{'seen': False, 'pos': None} for _ in range(self.num_targets)]
        for i in range(self.num_agents):
            x = state[i][13]
            y = state[i][14]
            r_cos = state[i][16]
            r_sin = state[i][17]
            theta = state[i][18]  # FOV (view width) theo định dạng base_env
            alpha = math.atan2(r_sin, r_cos)  # hướng quay
            agent_positions.append([x, y, theta, alpha])
            for j in range(self.num_targets):
                base = 22 + 5 * j
                seen_flag = bool(state[i][base + 4])
                if seen_flag:
                    target_positions[j]['seen'] = True
                    target_positions[j]['pos'] = (state[i][base], state[i][base + 1])
        return agent_positions, target_positions

    def _get_action_bounds_for_agent(self, i):
        """
        Trả về (low, high) cho agent i.
        low, high đều là np.array([delta_theta_min, delta_alpha_min])
        """
        if isinstance(self.base_env.action_space, gym.spaces.Tuple):
            box_i = self.base_env.action_space[i]
            lo, hi = box_i.low, box_i.high
        else:
            # fallback: nếu không phải Tuple thì coi như Box chung
            lo, hi = self.base_env.action_space.low, self.base_env.action_space.high
        return lo, hi
    
    def step(self, actions):
        """
        Một action = phân công nhiệm vụ (bitmask target cho từng agent).
        Trong step này, các agent sẽ chạy tối đa self.num_steps bước trong base_env.
        Nếu base_env done sớm thì dừng ngay.
        Trả về:
            next_state, mean_reward, done, info, steps_used
        """

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions)

        # Chuyển bitmask -> targets cho từng agent
        targets_per_agent = []
        for a in actions:
            targets = [t for t in range(self.num_targets) if (int(a) >> t) & 1]
            targets_per_agent.append(targets)

        culminate_reward = 0.0
        next_state = self._last_obs
        done = False 
        camera_infos = None

        # Rollout tối đa num_steps bước
        for current_step in range(1, self.num_steps + 1):
            agent_positions, target_positions = self._extract_position(next_state)
            deltas = []
            for i, agent_targets in enumerate(targets_per_agent):
                ax, ay, theta, alpha = agent_positions[i]
                seen_targets = [target_positions[j]['pos'] for j in agent_targets
                                if target_positions[j]['seen']]
                if seen_targets:
                    avg_x = sum(x for x, y in seen_targets) / len(seen_targets)
                    avg_y = sum(y for x, y in seen_targets) / len(seen_targets)
                    avg_pos = (avg_x, avg_y)
                    delta_alpha = compute_alpha((ax, ay), alpha, avg_pos)
                    delta_theta = compute_theta((ax, ay), theta, seen_targets)
                else:
                    lo, hi = self._get_action_bounds_for_agent(i)
                    delta_theta = random.uniform(float(lo[0]), float(hi[0]))
                    delta_alpha = random.uniform(float(lo[1]), float(hi[1]))
                deltas.append([delta_theta, delta_alpha])

            next_state, reward, done, camera_infos = self.base_env.step(deltas)
            self._last_obs = next_state
            culminate_reward += reward

            if done:
                break

        mean_reward = culminate_reward / current_step
        return next_state, mean_reward, done, camera_infos, current_step


if __name__ == "__main__":
    env = CoorEnv(num_steps=10)
    state = env.base_env.reset()

    done = False
    ep_reward = 0.0
    ep_steps = 0

    while not done and ep_steps < 50:  # ví dụ: 5 lần phân phối nhiệm vụ
        # random phân phối nhiệm vụ
        actions = np.random.randint(0, env.action_dim, size=(env.num_agents,))
        print("actions = ", actions)
        state, reward, done, info, used = env.step(state, actions)
        ep_reward += reward
        ep_steps += 1
        print(f"Commander step {ep_steps}: reward={reward:.4f}, used={used} base steps, done={done}")

    print(f"Episode finished: commander_steps={ep_steps}, total_reward={ep_reward:.4f}")