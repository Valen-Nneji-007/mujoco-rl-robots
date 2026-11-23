import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

XML_PATH = "/mnt/c/Users/Administrator/Documents/mujoco/mujoco_menagerie/unitree_go2/scene.xml"
FRAME_SKIP = 10
GAIT_PERIOD = 0.5   # seconds for one full trot cycle


class Go2Env(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, render_mode=None, **kwargs):
        utils.EzPickle.__init__(self, render_mode, **kwargs)
        observation_space = Box(-np.inf, np.inf, shape=(47,), dtype=np.float64)
        MujocoEnv.__init__(self, XML_PATH, FRAME_SKIP,
                           observation_space=observation_space,
                           render_mode=render_mode, **kwargs)
        self.action_space = Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        self.default_joints = self.model.key("home").qpos[7:].copy()
        self.foot_ids = [self.model.geom(n).id for n in ["FL", "FR", "RL", "RR"]]
        self.floor_id = self.model.geom("floor").id

        self.prev_action = np.zeros(12)
        self.air_time = np.zeros(4)
        self.phase = 0.0
        self.steps = 0

    # ---------- helpers ----------
    def _roll_pitch(self):
        w, x, y, z = self.data.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        return roll, pitch

    def _foot_contacts(self):
        c = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_id in pair:
                for k, fid in enumerate(self.foot_ids):
                    if fid in pair:
                        c[k] = True
        return c

    def _bad_contacts(self):
        bad = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self.floor_id in pair:
                other = con.geom2 if con.geom1 == self.floor_id else con.geom1
                if other not in self.foot_ids:
                    bad += 1
        return bad

    def _get_obs(self):
        height = self.data.qpos[2]
        roll, pitch = self._roll_pitch()
        body_vel = self.data.qvel[0:3]
        body_turn = self.data.qvel[3:6]
        joints = self.data.qpos[7:] - self.default_joints
        joint_vel = self.data.qvel[6:]
        clock = [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]
        return np.concatenate([[height, roll, pitch], body_vel, body_turn,
                               joints, joint_vel, self.prev_action, clock])

    # ---------- step ----------
    def step(self, action):
        action = np.clip(action, -1, 1)
        target = self.default_joints + 0.35 * action

        for _ in range(self.frame_skip):
            q = self.data.qpos[7:]
            qd = self.data.qvel[6:]
            torque = 60.0 * (target - q) - 1.5 * qd
            low, high = self.model.actuator_ctrlrange.T
            torque = np.clip(torque, low, high)
            self.do_simulation(torque, 1)

        self.steps += 1
        self.phase = (self.phase + self.dt / GAIT_PERIOD) % 1.0

        height = self.data.qpos[2]
        roll, pitch = self._roll_pitch()
        v = self.data.qvel[0]
        vz = self.data.qvel[2]
        joint_error = self.data.qpos[7:] - self.default_joints
        contacts = self._foot_contacts()
        bad = self._bad_contacts()

        # foot air time
        first_contact = (self.air_time > 0) & contacts
        self.air_time += self.dt

        reward = 2.0 * np.exp(-((v - 0.7) ** 2) / 0.1)
        reward += 1.0
        reward -= 0.5 * (roll ** 2 + pitch ** 2)
        reward -= 1.0 * vz ** 2
        reward -= 0.02 * np.sum((action - self.prev_action) ** 2)
        reward -= 20.0 * (height - 0.3) ** 2
        reward -= 1.0 * bad
        reward -= 0.3 * np.sum(joint_error ** 2)
        reward += 1.0 * np.sum((self.air_time - 0.2) * first_contact)
        reward -= 1.0 * self.data.qvel[1] ** 2
        reward -= 0.5 * self.data.qvel[5] ** 2

        # trot rhythm: FL+RR down in first half, FR+RL in second half
        pair_a_down = self.phase < 0.5
        want = np.array([pair_a_down, not pair_a_down, not pair_a_down, pair_a_down])
        reward += 0.5 * np.sum(contacts == want)

        self.air_time *= ~contacts
        self.prev_action = action.copy()

        fallen = height < 0.18 or abs(roll) > 0.8 or abs(pitch) > 0.8 or bad > 0
        if fallen:
            reward -= 5.0
        truncated = self.steps >= 1000

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, fallen, truncated, {}

    # ---------- reset ----------
    def reset_model(self):
        qpos = self.model.key("home").qpos.copy()
        qpos[7:] += self.np_random.uniform(-0.05, 0.05, size=12)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_action = np.zeros(12)
        self.air_time = np.zeros(4)
        self.phase = 0.0
        self.steps = 0
        return self._get_obs()