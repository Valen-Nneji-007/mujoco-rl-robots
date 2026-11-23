from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from go2_env import Go2Env

if __name__ == "__main__":
    env = make_vec_env(Go2Env, n_envs=6, vec_env_cls=SubprocVecEnv)
    model = PPO.load("go2_ppo_v7", env=env, device="cpu",
                     custom_objects={"learning_rate": 1e-4, "target_kl": 0.05})
    model.learn(total_timesteps=5_000_000, reset_num_timesteps=False)
    model.save("go2_ppo_v8")