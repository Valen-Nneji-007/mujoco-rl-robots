import gymnasium as gym
from stable_baselines3 import PPO

#env = gym.make("HalfCheetah-v5")
#model = PPO("MlpPolicy", env, verbose=1, device="cpu", target_kl=0.02)
#model.learn(total_timesteps=500_000)
#model.save("cheetah_ppo")

model = PPO.load("cheetah_ppo", device="cpu")
env = gym.make("HalfCheetah-v5", render_mode="human")
obs, info = env.reset()
for info in range(1000):
    action, info = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()