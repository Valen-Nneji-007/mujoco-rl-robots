from stable_baselines3 import PPO
from go2_env import Go2Env

model = PPO.load("go2_ppo_v8", device="cpu")
env = Go2Env(render_mode="human",
             default_camera_config={"trackbodyid": 1, "distance": 1.5, "elevation": -20})
obs, info = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()