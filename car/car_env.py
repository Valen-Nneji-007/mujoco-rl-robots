import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

class CarEnv(MujocoEnv):
    metadata = {"render_modes":["human", "rgb_array", "depth_array"], "render_fps":20}

    def __init__(self, render_mode=None):
        obs_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "car.xml"),
            frame_skip=5,
            observation_space=obs_space,
            render_mode=render_mode
        )
        self.prev_distance=0
        self.steps = 0

    def _get_obs(self):
        position = self.data.qpos
        velocity = self.data.qvel
        px,py,pz = position[0:3]
        qw, qx, qy, qz = position[3:7]
        ax, ay = position[7:9]

        # Velocity
        vx,vy,vz = velocity[0:3]
        tx, ty, tz = velocity[3:6]
        wx, wy = velocity[6:8]

        target = self.data.mocap_pos[0][:2]
        yaw = np.arctan2(2 * ( qw * qz + qx * qy ), 1 - 2 * (qy**2 + qz**2))

        # distance between the car and the target object
        dx = target[0] - px
        dy = target[1] - py

        # Going ahead and moving left
        ahead = np.cos(yaw) * dx + np.sin(yaw) * dy
        left = -np.sin(yaw) * dx + np.cos(yaw) * dy

        return np.array([ahead, left, vx, vy, tz, wx, wy])
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        car = self.data.qpos[0:2]
        ball = self.data.mocap_pos[0][:2]
        distance = np.linalg.norm(ball - car)
        reward = 10 * (self.prev_distance - distance)
        reward -= 0.001 * np.sum(np.square(action))
        self.prev_distance = distance
        self.steps += 1
        terminated = False
        if distance < 0.1:
            reward += 10
            terminated = True
        if abs(car[0]) > 2 or abs(car[1]) > 2:
            reward -= 10
            terminated = True
        truncated = self.steps >= 500
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {"distance": distance}
    
    def reset_model(self):
        self.steps = 0
        self.set_state(self.init_qpos, np.zeros(8))
        r = self.np_random.uniform(0.5, 1.5)
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.data.mocap_pos[0] = [r * np.cos(angle), r * np.sin(angle), 0.05]
        self.prev_distance = r
        return self._get_obs()






