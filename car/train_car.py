from car_env import CarEnv
from stable_baselines3 import PPO


env = CarEnv()
model = PPO("MlpPolicy", env, device="cpu",verbose=1, target_kl=0.02)
model.learn(total_timesteps=300_000)
model.save("car_ppo")

env = CarEnv(render_mode="human")
obs, info = env.reset()
for info in range(1000):
    action, info = model.predict(obs, deterministic = True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Final distance:", info["distance"])
        obs, info = env.reset()
env.close()


