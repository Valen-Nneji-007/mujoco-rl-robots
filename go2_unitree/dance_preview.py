import numpy as np
from go2_dance_env import Go2DanceEnv, ACTION_SCALE

env = Go2DanceEnv(render_mode="human")
obs, info = env.reset()
env.t = 0.0
for i in range(400):
    pose, _ = env._dance_target()
    action = pose / ACTION_SCALE
    obs, reward, terminated, truncated, info = env.step(action)
    roll, pitch = env._roll_pitch()
    print(i, "move:", info["move"],
          "height:", round(env.data.qpos[2], 3),
          "roll:", round(roll, 2), "pitch:", round(pitch, 2),
          "reward:", round(reward, 2))
    if terminated:
        print("--- Episode ended at step", i, "---")
        for k in range(env.data.ncon):
            c = env.data.contact[k]
            names = [env.model.geom(c.geom1).name, env.model.geom(c.geom2).name]
            if "floor" in names:
                print("Floor contact:", names)
        break
env.close()