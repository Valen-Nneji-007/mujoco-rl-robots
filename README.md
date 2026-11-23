# Reinforcement Learning for Quadruped Robot with MuJoCo

Teaching simulated robots to balance, hop, walk, trot and dance with reinforcement learning, from a single pendulum up to a **Unitree Go2** quadruped, all trained on a home laptop (GTX 1050, 8-core CPU).

Everything uses [MuJoCo](https://mujoco.org/) for physics, [Gymnasium](https://gymnasium.farama.org/) for the environments and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) for training.

<p align="center">
  <img src="docs/go2_trot.gif" width="420" alt="Unitree Go2 trotting in MuJoCo">
  <img src="docs/go2_dance.gif" width="420" alt="Unitree Go2 doing a lean, bow and sit-back dance">
</p>
<p align="center"><em>Left: trained Go2 trot. Right: four-move dance routine (shown at 2× speed).</em></p>

---

## Learning path

| # | Project | What it taught |
|---|---------|----------------|
| 1 | **Inverted Pendulum** | The basic RL loop: observe → act → reward → learn |
| 2 | **Double Pendulum** | Diagnosing a *policy collapse* (reward crashed from ~4,480 to ~149) and fixing it with `target_kl` |
| 3 | **Hopper** | First walking robot: 3 joints, reward for forward progress |
| 4 | **Walker2d** | Reward hacking: it learned to hop on both legs instead of walking |
| 5 | **HalfCheetah** | Episodes that never terminate; only the reward tells you how it's doing |
| 6 | **Ant / Humanoid** | Scaling up to 3D robots with many joints |
| 7 | **Custom 2-joint arm** | Writing my own MJCF robot and Gymnasium environment from scratch |
| 8 | **Custom wheeled car** | Free-floating bodies, quaternions → yaw, differential drive, success/failure termination |
| 9 | **Unitree Go2 walking** | A real robot model: PD control, contact sensing, gait shaping |
| 10 | **Unitree Go2 dance** | Pose tracking with a time-based target and smooth transitions |

---

## The Go2 story: what went wrong and how I fixed it

The Go2 took many iterations. Each version fixed a specific problem I saw in the simulation:

| Version | Problem I saw | Cause | Fix |
|---------|---------------|-------|-----|
| v1 | Dog only *pronked* (all four legs hopping together) | Alive bonus made standing safe; nothing punished bouncing | Target-speed reward, smaller alive bonus, vertical-velocity penalty |
| v2 | Training stuck: episodes always ~11 steps | Penalties made every step negative, so falling early was "optimal" | Alive bonus raised so staying up always pays |
| v3 | Still falling after 10 steps, even with zero actions | Menagerie's Go2 uses **torque motors**, but I was sending joint *angles* | Wrote my own **PD controller** (`torque = Kp·(target − q) − Kd·q̇`) |
| v4–v5 | Crouching, dragging back calves on the floor | Nothing rewarded height or punished non-foot contacts | Height reward, contact check that only allows feet on the floor |
| v6 | Legs still sagging (stand test: 0.27 m → 0.198 m) | PD gains too weak for the robot's weight | Stiffer gains (Kp 20 → 60), posture penalty, episode ends if a calf touches the floor |
| v7 | One back leg "limped" | No reason to use both rear legs | **Trot clock**: diagonal pairs (FL+RR, FR+RL) rewarded for alternating every 0.25 s, clock added to the observation |
| v8 | Short, shuffling steps; slight drift | Low target speed, small action range | Faster target speed, bigger action scale, sideways/yaw penalties |

Two small test scripts saved hours of wasted training:
- **`stand_test.py`**: sends zero actions and prints body height. Checks the robot can hold itself up at all.
- **`test_go2.py`**: sends random actions. Checks observation size, rewards and fall detection run without errors.

### Go2 walking environment (`go2_env.py`)

- **Observation (47):** height, roll, pitch, body linear and angular velocity, 12 joint angles and velocities, last action, trot-clock (sin/cos)
- **Action (12):** joint targets as offsets from the standing pose, turned into torques by a PD controller
- **Reward:** forward-speed tracking, alive bonus, foot air-time, trot rhythm, minus penalties for tilt, bouncing, jerky actions, height error, posture error, sideways/yaw motion and bad contacts
- **Training:** PPO, 6 parallel environments (`SubprocVecEnv`), ~10M steps

### Go2 dance (`go2_dance_env.py`)

Four moves, one second each, on repeat: **lean left, lean right, bow, sit back**.

Along the way I learned that:
- leaning only by changing leg length made the robot **roll over**: the hips have to counter-rotate so the legs stay vertical;
- jumping instantly between poses made it **swing over**: the target now **glides** to each new pose over 0.4 s;
- the full bow let the body **touch the floor**, so bow and sit-back use half the bend.

`dance_preview.py` plays the routine by sending poses straight to the motors (no brain), which is how I checked the poses were safe before training.

---

## Repository layout

```
.
├── Pendulum.py, double_pendulum.py, hopper.py, ...   # built-in Gymnasium robots
├── arm_project/       # custom 2-joint arm: arm.xml, arm_env.py, train_arm.py
├── car/               # custom wheeled robot: car.xml, car_env.py, train_car.py, watch_car.py
├── go2_unitree/
│   ├── go2_env.py           # Go2 walking environment
│   ├── train_go2_fast.py    # PPO training with 6 parallel environments
│   ├── watch_go2.py         # watch a trained model
│   ├── stand_test.py        # can the robot hold itself up?
│   ├── test_go2.py          # does the environment run?
│   ├── dance_poses.py       # the four dance poses
│   ├── go2_dance_env.py     # dance environment
│   ├── dance_preview.py     # play the dance without training
│   └── train_dance.py
└── docs/              # GIFs for this README
```

---

## Getting started

Tested on Ubuntu (WSL2) with Python 3.10.

```bash
python -m venv mujoco_env
source mujoco_env/bin/activate
pip install -r requirements.txt
```

The Go2 model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie):

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Then set `XML_PATH` at the top of `go2_unitree/go2_env.py` to point at `mujoco_menagerie/unitree_go2/scene.xml` on your machine.

**Run things:**

```bash
python go2_unitree/stand_test.py       # check the robot can stand
python go2_unitree/train_go2_fast.py   # train walking (a few hours on CPU)
python go2_unitree/watch_go2.py        # watch the result
python go2_unitree/dance_preview.py    # play the dance routine
```

> Training runs on the CPU (`device="cpu"`): the networks are small, so the CPU is faster than a GPU here. With 6 parallel environments I got about 470 steps per second.

---

## What I learned

- **The reward is the specification.** The agent learns exactly what you reward, not what you meant: hopping Walker2d, pronking Go2, the dog that fell on purpose.
- **Check the robot before training.** A 10-second stand test found a motor problem that hours of reward tuning never could.
- **Watch the training numbers.** `ep_len_mean` stuck at 11 meant the dog was falling every time, and `approx_kl` jumping to 3.85 revealed a policy collapse.
- **Change one thing at a time** when debugging, so you know what helped.

## Next steps

- Autonomous navigation for the Go2 with simulated range sensors and obstacle avoidance (rebuilding my internship project in simulation)
- Unitree G1 humanoid: walking, then dancing with motion imitation

## Acknowledgements

- [MuJoCo](https://github.com/google-deepmind/mujoco) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Unitree Go2 model) by Google DeepMind
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) by the Farama Foundation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
