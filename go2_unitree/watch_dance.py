from stable_baselines3 import PPO
from go2_dance_env import Go2DanceEnv

model = PPO.load("go2_dance_v1", device="cpu")
env = Go2DanceEnv(render_mode="human")
obs, info = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()