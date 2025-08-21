import mate
import time
env = mate.make('MultiAgentTracking-v0')
env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
env.seed(0)
print(env.num_teammates)
print(env.num_opponents)
print(env.action_space)
print("------------------")
print(env.action_space[0].high)
time.sleep(10000)
done = False
camera_joint_observation = env.reset()
while not done:
    camera_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    print(camera_joint_action)
    print("type = ", type(camera_joint_action[0]))
    print("obs = ", camera_joint_observation)
    time.sleep(100)
    camera_joint_observation, camera_team_reward, done, camera_infos = env.step(camera_joint_action)

