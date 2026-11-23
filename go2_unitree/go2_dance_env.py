import numpy as np
from gymnasium.spaces import Box
from go2_env import Go2Env
from dance_poses import DANCE, MOVE_TIME

ACTION_SCALE = 0.8   # bigger than walking, so the legs can reach the 0.6 bend


class Go2DanceEnv(Go2Env):

    def __init__(self, render_mode=None, **kwargs):
        self.t = 0.0                      # dance clock, in seconds
        Go2Env.__init__(self, render_mode=render_mode, **kwargs)
        # 3 + 3 + 3 + 12 + 12 + 12 + 12 (target pose) + 2 (move clock) = 59
        self.observation_space = Box(-np.inf, np.inf, shape=(59,), dtype=np.float64)

        # ---------- which pose should the dog be in right now? ----------
    def _dance_target(self):
        n = len(DANCE)
        move = int(self.t / MOVE_TIME) % n
        progress = (self.t % MOVE_TIME) / MOVE_TIME     # 0 to 1 within the move
        prev = DANCE[(move - 1) % n]
        blend = min(progress / 0.4, 1.0)                 # glide, then hold
        pose = prev + blend * (DANCE[move] - prev)
        return pose, progress

    # ---------- what the dog sees ----------
    def _get_obs(self):
        height = self.data.qpos[2]
        roll, pitch = self._roll_pitch()
        pose, progress = self._dance_target()
        clock = [np.sin(2 * np.pi * progress), np.cos(2 * np.pi * progress)]
        return np.concatenate([
            [height, roll, pitch],
            self.data.qvel[0:3],
            self.data.qvel[3:6],
            self.data.qpos[7:] - self.default_joints,
            self.data.qvel[6:],
            self.prev_action,
            pose,
            clock,
        ])

    # ---------- one step ----------
    def step(self, action):
        action = np.clip(action, -1, 1)
        target = self.default_joints + ACTION_SCALE * action

        for _ in range(self.frame_skip):
            q = self.data.qpos[7:]
            qd = self.data.qvel[6:]
            torque = 60.0 * (target - q) - 1.5 * qd
            low, high = self.model.actuator_ctrlrange.T
            self.do_simulation(np.clip(torque, low, high), 1)

        self.t += self.dt
        self.steps += 1

        pose, _ = self._dance_target()
        joint_error = (self.data.qpos[7:] - self.default_joints) - pose
        height = self.data.qpos[2]
        roll, pitch = self._roll_pitch()
        contacts = self._foot_contacts()
        bad = self._bad_contacts()
        vx, vy = self.data.qvel[0], self.data.qvel[1]
        yaw_rate = self.data.qvel[5]

        reward = 3.0 * np.exp(-np.sum(joint_error ** 2) / 0.5)   # match the pose
        reward += 0.5                                            # alive
        reward += 0.25 * np.sum(contacts)                        # keep feet down
        reward -= 2.0 * (vx ** 2 + vy ** 2)                      # don't walk away
        reward -= 0.5 * yaw_rate ** 2                            # don't spin
        reward -= 0.02 * np.sum((action - self.prev_action) ** 2)  # smooth
        self.prev_action = action.copy()

        fallen = height < 0.15 or abs(roll) > 1.0 or abs(pitch) > 1.0 or bad > 0
        if fallen:
            reward -= 5.0
        truncated = self.steps >= 1000

        if self.render_mode == "human":
            self.render()
        move = int(self.t / MOVE_TIME) % len(DANCE)
        return self._get_obs(), reward, fallen, truncated, {"move": move}

    # ---------- new episode ----------
    def reset_model(self):
        Go2Env.reset_model(self)
        # start at a random point in the routine, so it practises every move
        self.t = self.np_random.uniform(0, len(DANCE) * MOVE_TIME)
        return self._get_obs()