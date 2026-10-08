# HW2 learning notes: robot control and MDPs

One task runs through the whole homework: **move the robot's hand (the red `ee_site` point) to a target**.
The three exercises solve it in three stages:

| | Question | Who decides how | What you see |
|---|---|---|---|
| ex1 | Which joint angles put the hand there? | Math: inverse kinematics | The arm *teleports* to 16 points on an ∞ curve |
| ex2 | How do motors actually move it there smoothly? | Hand-written rules: spline + PID | The arm glides along the ∞ curve |
| ex3 | Can it learn to reach without hand-written rules? | A neural network trained by RL (PPO) | The arm reaches random green targets |

The core idea of the course is the ex2 vs ex3 contrast: **a controller designed by humans** versus **a controller learned from rewards**.

---

## 1. Building blocks

### The robot (SO-100, 6 joints)

| Joint | Plain meaning |
|---|---|
| Rotation | Turns the whole arm left and right |
| Pitch | Shoulder, lifts the arm |
| Elbow | Bends the arm in the middle |
| Wrist_Pitch | Tilts the hand |
| Wrist_Roll | Twists the hand |
| Jaw | Opens and closes the gripper |

### Two ways to describe the arm
- **Joint space**: the 6 joint angles, `data.qpos`.
- **Task space** (workspace): where the hand is in the room, as x, y, z.
- **Forward kinematics**: angles → hand position. Easy geometry.
- **Inverse kinematics (IK)**: desired hand position → angles. Hard, so it's solved iteratively (ex1).

### MuJoCo, the physics simulator
A physics engine: you describe the robot in XML (`so101_gym/assets/`), and MuJoCo computes how it moves under motor forces, gravity, contacts and joint limits.

- `model` is the fixed description from the XML: masses, joint limits, sizes. It never changes.
- `data` is the current state, which changes every step:
  - `data.qpos`: joint angles
  - `data.qvel`: joint speeds
  - `data.ctrl`: motor commands
  - `data.site("ee_site").xpos`: hand position
  - `data.mocap_pos[0]`: the green target ball. A "mocap" body has no physics; code just moves it.
- `mj_step(model, data)` **advances time by 0.002 s** (500 Hz) and runs the physics: force → acceleration → velocity → position.
- `mj_forward(model, data)` **recomputes positions from the current angles without advancing time**. No physics.

Two robot files differ only in what `data.ctrl` means:

| File | `data.ctrl` means | Used in |
|---|---|---|
| `so100_torque_ctrl.xml` | Motor force (torque) | ex2: your PID computes it |
| `so100_pos_ctrl.xml` | Target angle; a built-in motor controller supplies the force | ex1, ex3 |

### Running things on macOS
```bash
cd hw2_robot_control_mdps
source .venv/bin/activate          # Homebrew Python 3.12; mjpython needs a framework build
mjpython scripts/<viewer script>.py  # anything that opens a MuJoCo window
python scripts/train.py              # no window needed
```
Always run from the `hw2_robot_control_mdps` folder; the scripts find `env/` and `exercises/` relative to it.

---

## 2. The key difference: what the loop writes into MuJoCo

| | Written each loop | Does time advance? |
|---|---|---|
| ex1 | Joint angles directly: `data.qpos[:] = answer` | No → "teleport" |
| ex2 | Motor forces: `data.ctrl = PID(error)` | Yes, 0.002 s per loop |
| ex3 | Target angles: `data.ctrl = process_action(network(obs))` | Yes, 0.1 s per action (50 physics steps) |

---

## 3. ex1: inverse kinematics (`exercises/ex1.py`)

**Execution path:** `mjpython scripts/inverse_kinematics.py`
```python
keypoints = build_keypoints()                         # 16 points on the ∞ curve
while viewer.is_running():
    target_qpos = ik_track(model, data, "ee_site", kp)   # ① solve IK ("think")
    data.qpos[:] = target_qpos                           # ② overwrite the angles: teleport
    mujoco.mj_forward(model, data)                       # ③ recompute hand position, time stays 0
    viewer.sync(); next keypoint every 0.5 s
```
`mj_step` is never called, so physics never runs. That's why the arm jumps from point to point.

**What I implemented**
- `get_lemniscate_keypoint`: y = a·cos t / (1+sin²t), z = a·cos t·sin t / (1+sin²t).
- `build_keypoints`: `t = linspace(0, 2π, count, endpoint=False)`. Excluding 2π avoids a duplicate start point. Rows are `[x_offset, y, z + z_offset]`.
- `ik_track`: repeat up to 2000 times:
  1. error = target − hand position; stop if below 1 mm.
  2. Jacobian `J` = "if each joint turns a tiny bit, how does the hand move" (`mj_jacSite`).
  3. Damped least squares: `qdot = Jᵀ (J Jᵀ + λI)⁻¹ e`, solved with `np.linalg.solve`. The 6D error is `[pos_gain·err, 0, 0, 0]` because orientation is ignored. The damping λ keeps steps bounded near singular poses, like a fully stretched arm.
  4. `qpos += qdot · dt`.

  Finally restore the original `qpos` and return the answer. IK only "thinks"; it doesn't move the robot.

**What I measured**
- Default settings: all 32 solves over 2 laps landed within 5 mm (worst 2.8 mm).
- Starting from qpos = 0, IK converges to a *mirrored* pose that breaks joint limits (elbow up to 1.65 rad past its limit). The skeleton's solver ignores limits, and teleporting bypasses MuJoCo's limit enforcement. Naively clamping to limits made it reach only 10/16 points.

| Width a | Points reached (of 16) |
|---|---|
| 0.25 (default) | 16 |
| 0.35 | 12 |
| 0.45 | 8 |
| 0.60 | 8 |

| IK step dt | Points reached (of 16) |
|---|---|
| 0.005 | 0 (too slow, runs out of iterations) |
| 0.05 | 15 |
| 0.1 (default) | 16 |
| 0.5 | 16 |
| 1.0 | 12 |
| 2.0 | 0 (overshoots and diverges) |

**Theory answers**
1. Bigger `a`: some points leave the reachable workspace or sit near singular poses, so IK stops converging.
2. `dt` is the step size: too small runs out of iterations, too large overshoots and diverges.
3. Numerical vs analytical IK: numerical works for any robot and is easy to write, but it is iterative, depends on the initial guess, can land on odd solutions, and has no convergence guarantee. Analytical IK is fast, exact and gives all solutions, but must be derived per robot.
4. Versus state-of-the-art solvers (Pinocchio, mink): ours ignores joint limits, orientation, collisions and velocity limits, which those solvers handle as optimization constraints.

---

## 4. ex2: trajectory + PID (`exercises/ex2.py`)

**Execution path:** `mjpython scripts/pid_control.py`
```python
model = load(TORQUE_CTRL_XML)                 # ctrl now means motor force
waypoints = 16 segments × quintic spline (5 points each) = 80 waypoints
mujoco.set_mjcb_control(pid_callback)         # "before each physics step, ask pid_callback for motor forces"
while viewer.is_running():                    # one loop = 0.002 s
    target_qpos = ik_track(..., waypoint)     # ex1 gives target angles
    error history ← target_qpos − data.qpos
    mujoco.mj_step(model, data)               # inside: pid_callback → data.ctrl = pid_control(history)
                                              #         then physics moves the arm a tiny bit
    if hand within 3 cm of waypoint: next waypoint
```
The loop **never writes `data.qpos`**. It only sets forces, and physics moves the arm, so the motion is continuous.

**Yes, ex2 is all hand-written rules:**
- Route rule, the quintic time scaling `f(s) = 10s³ − 15s⁴ + 6s⁵`. Progress at s = 0, 0.25, 0.5, 0.75, 1 is 0%, 10%, 50%, 90%, 100%: slow, fast, slow. Velocity and acceleration are zero at both ends, so starts and stops are smooth.
- Force rule, PID: `ctrl = Kp·P + Ki·I + Kd·D` with Kp = 150, Ki = 0, Kd = 0.01.
  - P = latest error: push harder when far away.
  - I = sum(errors) × dt: keep adding push if it keeps falling short.
  - D = (e[-1] − e[-2]) / dt: brake when approaching fast. It's 0 with only one sample.

**PID intuition, 1D block pulled back by a constant force of 2, target 1.0**

| Controller | Behavior | Ends at |
|---|---|---|
| P, Kp = 10 | Oscillates 0 ↔ 1.6 forever | never settles |
| P, Kp = 100 | Faster, still oscillates forever | never settles |
| PD, Kp = 100, Kd = 20 | No overshoot, smooth stop | 0.98 (steady-state error = 2/100) |
| PID, + Ki = 50 | Small overshoot, then settles | 1.004 |

**Theory answers**
1. Increasing Kp: faster but overshoots and oscillates; at very high gains it never settles or goes unstable. In the sim, Kp ≥ 1000 never settled.
2. Kd opposes the rate of change of the error, so it brakes the arm before the target and damps the oscillation.
3. Ki is needed under constant disturbances (gravity, friction, payload): P alone settles short of the target, and the integral accumulates that error until it's removed.

Side note: the script keeps only the last 10 errors, so its "integral" covers 0.02 s and acts almost like a small extra P term.

---

## 5. ex3: reinforcement learning (`exercises/ex3.py`)

### What an MDP is
A Markov decision process is the loop of any "learn by trying" problem:
```
look at the situation (state) → choose an action → the world changes → get a score (reward) → repeat
```
- **Policy**: the rule from "what I see" to "what I do". Here, a small neural network.
- **Goal**: maximize total reward.
- **Markov**: the current state contains everything needed to decide; no memory of the past is required.

| MDP piece | In hw2 |
|---|---|
| State / observation | `get_obs`: 6 joint angles + hand position + hand quaternion + target position = 16 numbers, all in the robot's base frame |
| Action | 6 numbers in [−1, 1]; `process_action` maps −1 → joint low limit, 0 → midpoint, +1 → high limit |
| Transition | MuJoCo runs 50 physics steps (0.1 s); the policy acts at 10 Hz |
| Reward | `compute_reward`: `exp(−2·error)` + 1 if error < 5 mm |
| Episode | 100 actions (10 s), then reset |
| Reset | `reset_robot`: default pose + uniform(−0.5, 0.5) noise; `reset_target_position`: base + uniform offset in x [0.2, 0.4], y [−0.2, 0.2], z [0.1, 0.4] |

### Why these design choices
- **Random starts and targets**: the policy learns "reach anything" instead of memorizing one motion.
- **Base frame instead of world frame**: the policy cares where the target is *relative to the robot*, so it still works if the robot moves.
  - Position: `p_base = R_baseᵀ (p_world − base_pos)`
  - Orientation: `q_ee_base = normalize(q_base⁻¹ ⊗ q_ee_world)`. MuJoCo quaternions are (w, x, y, z), and multiplication order matters.
- **Dense + sparse reward**: dense ("closer is better") gives a learning signal everywhere, so learning can start. Sparse (+1 within 5 mm) pushes for precise arrival instead of hovering nearby.

### Training execution path: `python scripts/train.py`
```python
envs  = SubprocVecEnv([16 robots])        # 16 copies collect experience in parallel
model = PPO("MlpPolicy", envs)            # random network, knows nothing yet
model.learn(...)                          # 500 rounds of:
#   1. collect: each robot takes 2048 steps
#        network(obs) → action (+ exploration noise)
#        env.step(action):  data.ctrl = process_action(action)
#                           50 × mj_step
#                           reward = compute_reward(distance)
#                           obs = get_obs(...)
#        every 100 steps: reset (random pose and target)
#   2. update on the 32,768 collected steps: actions that scored better than
#      expected become more likely, worse ones less likely (PPO)
# a checkpoint is saved every 50 rounds → logs/so100_tracking/so100_tracking_1/model_N.zip
```
500 rounds ≈ 16.4M steps, about 45–60 min on CPU (M-series Mac, 16 envs).

### Evaluation execution path
`mjpython scripts/evaluate_rand_targets.py --load_run=1 --checkpoint=500`

Load the network. Every 50 physics steps: `obs → rl_model.predict(obs, deterministic=True) → data.ctrl`. Each episode lasts 2 s, then the final distance is printed. After 10 episodes the average is printed; it must be **< 0.05 m**.

### My results (run 1)

| Checkpoint | 10-episode average | 100-episode average | Episodes < 0.05 |
|---|---|---|---|
| 50 | 0.052 | 0.078 | 39% |
| 100 | 0.024 | 0.022 | 98% |
| 200 | 0.010 | 0.010 | 99% |
| 300 | 0.006 | 0.008 | 99% |
| 500 | 0.006 | 0.007 | 98% |

About 1–2 targets in 100 still aren't reached within 2 s (worst ≈ 0.09 m).

### Bonus idea (not done)
`mjpython scripts/evaluate_trajectory.py --load_run=1 --checkpoint=500` makes the policy follow the ∞ curve. It was trained only on *static* targets, so expect jitter and lag compared with PID. Things to try:
- penalize large changes between actions, for smoothness;
- add target velocity or the next target to the observation;
- train with moving targets;
- tune PPO's `gamma` and `ent_coef`.

---

## 6. Experiments to build intuition yourself

1. `mjpython scripts/interactive.py`: drag the joint sliders and see what each joint does.
2. Print `data.time` after the `mj_forward` line in `inverse_kinematics.py`: it stays 0. Do the same after `mj_step` in `pid_control.py`: it grows by 0.002 each loop. That is "teleport" versus real motion.
3. Set `width=0.5` in `build_keypoints` and watch IK fail.
4. Set Kp to 2000 in ex2 and watch the arm oscillate.
5. Return only the sparse reward in ex3 and train 100 iterations: learning becomes much harder.
6. Redo the exercises from the blank TODOs (back up your versions first):
   ```bash
   mkdir -p ~/hw2_reference && cp exercises/ex*.py ~/hw2_reference/
   git checkout -- exercises/ex3.py
   ```

## 7. Video checklist (total < 5:30 with bonus, < 4:30 without)
- ex1 (< 2 min): robot tracking the keypoints + 4 theory answers.
- ex2 (< 2 min): robot moving between waypoints + 3 theory answers.
- ex3 (< 30 s): random-target tracking **with the terminal error printouts visible**. Bonus: +1 min.
