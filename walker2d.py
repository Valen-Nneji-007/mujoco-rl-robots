import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Walker2d-v5")
model = PPO("MlpPolicy", env, verbose=1, device="cpu", target_kl=0.02)
model.learn(total_timesteps=500_000)
model.save("walker_ppo")

env = gym.make("Walker2d-v5", render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()