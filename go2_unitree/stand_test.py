import numpy as np
from go2_env import Go2Env

env = Go2Env()
obs, info = env.reset()
print("gain type:", env.model.actuator_gaintype[:3], "bias type:", env.model.actuator_biastype[:3])
for i in range(100):
    obs, reward, terminated, truncated, info = env.step(np.zeros(12))
    print(i, "height:", round(obs[0], 3), "fallen:", terminated)
    if terminated:
        break
env.close()