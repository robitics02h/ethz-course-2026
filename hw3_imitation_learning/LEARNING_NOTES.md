# HW3 notes

The three exercises build on the same task: pick up a cube and put it in a bin. Ex1 learns from demonstrations, Ex2 adds corrections when the policy struggles, and Ex3 chooses a target among several colored cubes.

Our demonstrations came from a scripted expert in the simulator. The assignment describes human teleoperation, so our data collection method differs from that workflow.

## How the policy works

A policy is a function that takes the robot's current observations and predicts an action. Ours uses positions, joint angles, and gripper state as input. It does not use camera images.

Each action has four values:

```text
[delta_x, delta_y, delta_z, gripper_angle]
```

The first three move the hand's control target. The last sets the gripper opening. The network predicts four actions at once, executes them, then reads the state again. At 10 Hz, that means a new prediction every 0.4 seconds.

Behavior cloning (BC) trains this network on examples of what the expert did. Given a recorded state, it tries to predict the expert's next four actions. Training reduces the difference between those predictions and the recorded commands.

There is no reward for finishing the task during BC training. That explains a recurring problem: a model can have lower prediction error and still fail to release the cube. We have to run it in the simulator to see whether it actually works.

The scripted expert has explicit grasp, move, and release stages. The learned policy gets those behaviors from the data; it does not call the expert during evaluation.

## Ex1: learning to pick and place

Three changes mattered:

- **Fix the action labels.** We initially used the hand's observed movement. But the simulator expects movement of the control target, which can differ when the hand hits something or lags behind. We switched to the actual recorded commands.
- **Improve the demonstrations.** The expert sometimes caught its open gripper on the obstacle. Closing it while passing the obstacle made the path more reliable. We also varied the starting positions more and collected 240 successful demonstrations.
- **Check complete runs.** A one-step model had lower prediction error but often failed to release. We kept the four-step model. Other settings differed too, so this does not prove four steps are always better.

The final model passed **100/100 official episodes** and **200/200 separate test episodes**. The latter also checked that the gripper opened and moved away while the cube stayed in the bin.

[Experiment details](artifacts/ex1_improvement/run_info.json). The local policy video is at `artifacts/ex1_improvement/policy_preview.mp4`.

## Ex2: learning from the policy's mistakes

Ex2 moves the obstacle into less familiar positions. The Ex1 model dropped to **17/100** on the official test. It had learned the original layouts well, but struggled after the obstacle moved.

DAgger collects examples from states the policy itself reaches:

1. Let the current policy act.
2. Have an expert take over and demonstrate what to do from there.
3. Merge those corrections with the original demonstrations and train again.

We collected 24 automatic takeovers, balanced across left, center, and right obstacle positions. The scene was not reset at takeover. We combined these corrections with the 240 Ex1 demonstrations and trained the same network architecture from scratch.

The result was **79 successful episodes out of 100**, corresponding to an **official score of 80/100**. Some transport and release failures remained. We did not run a separate final holdout for this version.

Successful demonstrations from a fresh reset teach the normal path. DAgger also teaches what to do after the policy has already gone off that path.

[Experiment details](artifacts/ex2/run_info.json)

## Ex3: choosing the right cube

The first model received the raw state and color goal and scored **15/100**. The biggest improvement came from how we represented its input.

We used the requested color to select the target cube's initial position, then gave the network positions relative to the hand and bin. This made the geometry of the task easier to learn. With the same 60 demonstrations, the official result rose to **98/100**.

Next, we added 240 ordinary, complete demonstrations covering more starting layouts. Fine-tuning on the resulting 300-demonstration dataset produced the model we kept: **99/100 officially**, and **996/1000 on a separate strict test**. Later batches were around 99–99.4%, so 99.6% is one measured result, not a guarantee.

[Input representation experiment](artifacts/ex3/improvement_run_info.json) · [300-demonstration experiment](artifacts/ex3/coverage/run_info.json)

We also tried several ways to improve the remaining failures:

| Attempt | What happened |
| --- | --- |
| Execute only one or two of the four predicted actions | Much worse with the existing model. Kept four. |
| Add recovery demonstrations | Tried six corrections, fine-tuning at a lower learning rate, 33 earlier takeovers, and a smaller share of recovery data. None gave a reliable overall improvement. |
| Continue training on the old data | Small gains in one round, but insufficient evidence to replace the model then. Later rounds also introduced failures. |
| Sample the transport stage more often | No better than continuing training without the extra weighting. |
| Add 60 release-and-retreat demonstrations | More failures overall. |
| Make the gripper strictly open or closed | Both thresholding and a trained classifier performed worse. |
| Limit the gap between control target and actual hand | Did not fix the failures. This was a simulator diagnostic, not a submitted policy change. |
| Add 216 demonstrations of difficult layouts | Fixed three failures but introduced three others; official result fell to 98/100. |

Detailed settings and results are in the [experiment reports](artifacts/ex3/). We stopped experimenting and kept the 300-demonstration model.

## What I would remember

Better input representation and better coverage of ordinary scenes gave us the clearest gains. Adding difficult examples was less predictable: it could fix a rare failure while changing behavior that already worked.

A smaller training loss was not enough to choose a model. We needed complete runs on matching scenes, followed by fresh scenes for confirmation. Training and validation were split by whole episodes so nearby frames from one demonstration could not appear in both.

For the final versions, use [model.py](hw3/model.py) with the saved files below. A `.pt` file stores the trained weights and settings; it is not a recording. Checkpoints, `.hwresult` files, datasets, and videos are local outputs excluded from Git. The repository keeps the code and experiment summaries.

| Exercise | Local model | Official result | Local result file |
| --- | --- | --- | --- |
| Ex1 | `ex1.pt` | 100/100 | `ex1_result.hwresult` |
| Ex2 | `ex2.pt` | 80/100 score; 79 successes | `ex2_result.hwresult` |
| Ex3 | `ex3.pt` | 99/100 | `ex3_result.hwresult` |
