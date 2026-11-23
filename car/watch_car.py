from stable_baselines3 import PPO
from car_env import CarEnv

model = PPO.load("car_ppo", device="cpu")
env = CarEnv(render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Final distance:", info["distance"])
        obs, info = env.reset()

env.close()