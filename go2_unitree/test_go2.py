from go2_env import Go2Env

env = Go2Env()
obs, info = env.reset()
print("Observation size:", obs.shape)
print("Action space:", env.action_space)

for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(i, "reward:", round(reward, 3), "fallen:", terminated)
    if terminated or truncated:
        obs, info = env.reset()
env.close()