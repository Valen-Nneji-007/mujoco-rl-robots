from go2_env import Go2Env
from stable_baselines3 import PPO

env = Go2Env()
model = PPO("MlpPolicy", env, verbose=1, device='cpu', target_kl = 0.02)
model.learn(total_timesteps=4_000_000)
model.save("go2_ppo_v6")

env = Go2Env(render_mode="human")
obs, info = env.reset()
for info in range(1000):
    action, info = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print("forward speed:", info["forward_speed"])
    if terminated or truncated:
        obs, info = env.reset()
        
env.close()