import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("InvertedPendulum-v5")
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=50_000)
model.save("pendulum_ppo")

env = gym.make("InvertedPendulum-v5", render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _= env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()