# R1 Lite FastWAM inference

## Absolute-arm stack-bowl checkpoint (step 5000)

Start the workstation server with the checkpoint, resolved inference config,
dataset statistics, and cached singular prompt embedding:

```bash
cd /home/spate308/Documents/FastWAM
bash experiments/r1lite/run_fastwam_absolute_server.sh
```

This checkpoint predicts absolute targets for all 12 arm joints and absolute
targets for both grippers. Absolute mode is now the client default, and the
client/server contract check rejects a relative/absolute mismatch before any
returned action can be executed. Launch the neutral-left client as usual, or
state the mode explicitly:

```bash
bash experiments/r1lite/run_r1lite_client_neutral_left.sh \
  --server http://WORKSTATION_WIRED_IP:8000 \
  --prompt 'stack the bowl' \
  --action-mode absolute \
  --execute
```

`serve_fastwam_r1lite.py` runs on the GPU workstation. It does all FastWAM-specific work: camera composition, `[-1, 1]` image normalization, proprioception normalization with the run's `dataset_stats.json`, action inference, and action de-normalization.

`inference_r1lite_fastwam.py` runs on the R1 Lite computer. It uses the existing ROS 2 topics:

- feedback: `/hdas/feedback_arm_left`, `/hdas/feedback_arm_right`, `/hdas/feedback_gripper_left`, `/hdas/feedback_gripper_right`;
- commands: `/motion_target/target_joint_state_arm_left`, `/motion_target/target_joint_state_arm_right`, `/motion_target/target_position_gripper_left`, `/motion_target/target_position_gripper_right`;
- head camera: `/hdas/camera_head/left_raw/image_raw_color/compressed`.

Raw wrist image topics are the default because they match the existing r1lite inference client. Use `--wrist-transport compressed` if the robot stack only exposes the `/compressed` variants.

## Action contracts

```text
state:  [left_arm_6, right_arm_6, left_gripper, right_gripper]

absolute model:
action: [left_arm_target_6, right_arm_target_6,
         left_gripper_target, right_gripper_target]

legacy relative model:
action: [delta_left_arm_6, delta_right_arm_6,
         left_gripper_target, right_gripper_target]
```

The server de-normalizes actions, clips every dimension to the loaded
dataset's `global_min`/`global_max` action bounds, and reports its action
contract. In absolute mode the client publishes the returned 14-D target
directly after an additional physical-safety-limit check; it does not add live
feedback. Relative mode remains available explicitly for legacy checkpoints.

For a legacy relative checkpoint, the associated R1 Lite training config must
use 14-D data with arm actions already represented as deltas and absolute
gripper actions. Include:

```yaml
delta_action_dim_mask:
  default: [true, true, true, true, true, true,
            true, true, true, true, true, true,
            false, false]
```

The mask does not create deltas. Convert only arm actions to deltas in the training-data conversion path.

## FastWAM camera preprocessing

The server first matches R1 Lite collection framing, then follows FastWAM's RobotWin layout:

```text
live head        -> center-crop 640x640 -> bilinear 320x256
live left wrist  -> native 640x360 -> bilinear 160x128
live right wrist -> native 640x360 -> bilinear 160x128

head above [left wrist | right wrist] -> RGB 320x384 -> CHW float [-1, 1]
```

The collection-size crops happen on the workstation, before the FastWAM mosaic, so live observations match the demonstrations.

## One-time model preparation

Run this in the FastWAM environment before training or inference:

```bash
cd /home/spate308/Documents/FastWAM
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The workstation launcher sets `DIFFSYNTH_MODEL_BASE_PATH` to this repository's `checkpoints/` directory automatically when it is not already set.

## One-time prompt embedding

The server uses cached text context so the T5 encoder does not need to remain loaded during inference. Generate the embedding for the exact task prompt once:

```bash
cd /home/spate308/Documents/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/precompute_text_embeds.py \
  --config-name right_arm_stack_bowl_config \
  +override_instruction="stack the bowl" \
  +overwrite=true
```

This writes the cache under `data/text_embeds_cache/r1_lite_tasks`, as configured by `right_arm_stack_bowl_config.yaml`. The prompt passed to the robot must match the precomputed task string. If the GPU runs out of memory while generating the embedding, run this one-time command with `CUDA_VISIBLE_DEVICES=""` to use CPU.

## Run

Install FastAPI and Uvicorn in the FastWAM environment if necessary:

```bash
python -m pip install 'fastapi>=0.110' 'uvicorn[standard]>=0.27'
```

On the workstation:

```bash
bash experiments/r1lite/run_fastwam_server.sh \
  --checkpoint /path/to/r1lite_fastwam.pt \
  --dataset-stats /path/to/r1lite_dataset_stats.json \
  --config configs/right_arm_stack_bowl_config.yaml \
  --text-embedding-cache-dir data/text_embeds_cache/r1_lite_tasks \
  --host 0.0.0.0 --port 8000
```

`--config` must be the FastWAM Hydra config associated with the R1 Lite checkpoint. Its processor dimensions and normalization statistics must match the training run.

Each request uses publisher ROS timestamps with a maximum 50 ms span across all
seven streams. The oldest selected sample must also be at most 250 ms old at
selection (`--max-observation-age-ms`), and the head frame must be newer than the
previous request. Publishers must provide valid capture stamps on the same ROS
clock; zero stamps are rejected with a warning. The client waits if no qualifying
bundle exists. Selection timestamps and measured spans are logged in the robot's
`run.hdf5` demo events. These limits apply before the blocking model request; they
do not eliminate inference latency.

Execution events and action summaries are recorded on the robot. The unused
server `/runs/event` and `/runs/action-step` endpoints have been removed;
`/runs/start`, `/infer`, `/metadata`, and `/healthz` remain available.
After each successful model call, the workstation saves the decoded native
head and wrist images under `server_observations/run_N/demo_M/chunks/chunk_K/`.

On the robot, source ROS 2 and start in dry-run mode:

```bash
bash experiments/r1lite/run_r1lite_client.sh \
  --server http://WORKSTATION_IP:8000 \
  --prompt 'stack the bowl' \
  --policy-config right_arm_stack_bowl
```

At startup the client asks for the number of demos, then creates one robot-local
session at:

```text
rollouts/r1lite/<policy_config>/<YYYYMMDD>/run_<HHMMSS_microseconds>/run.hdf5
```

Each demo begins with a valid synchronized three-camera observation. A single
background writer stores continuous JPEG camera frames, measured states, held
targets, executed model actions, feedback, events, command publications, and
the exact per-query JPEG payloads in the HDF5. The collection viewer can read
the `data/demo_N` camera and state arrays. Its `actions` array is target minus
measured state; the original model actions are stored separately under
`rollout/steps`. Live rollout HDF5 files cannot be passed to `--replay-hdf5`.

When Enter stops a demo, the robot starts resetting immediately while the
writer closes and checkpoints the demo. After `Reset finished.`, the client
prompts for an integer accuracy score from 1 to 100 (input typed during reset
is ignored); only 100 counts as success. The score and reset result
are included in a second checkpoint. The next demo starts after reset and both
checkpoints finish. At the end, the client prints successes divided by the
number of demos requested at startup, and stores those totals in `run.hdf5`.

During recording, `run.inprogress` is the active file and
`run.checkpoint.tmp` is used for an atomic replacement of `run.hdf5`. Only
`run.hdf5` is intended for the viewer. A power loss may lose the active demo;
previously checkpointed demos remain in `run.hdf5`. The workstation still
saves its independent decoded native images under `server_observations/`.

After verifying state order, gripper units, target limits, and live camera topics:

```bash
bash experiments/r1lite/run_r1lite_client.sh \
  --server http://WORKSTATION_IP:8000 \
  --prompt 'stack the bowl' --execute --control-hz 15
```

The client executes at 15 Hz, matching the demonstrations' physical capture rate, and by default executes the complete action horizon returned by the trained server (32 actions for this checkpoint). Pass `--replan-steps N` only to deliberately use a shorter receding horizon.

The client initially waits without querying the server or moving the robot. Press Enter to start inference. Press Enter again to stop, hold the latest measured pose, and run the shared arm/gripper/torso reset back to `initial_robot_position.json`. Inference takes the arm/gripper hold back at that reset target. After scoring and checkpointing, press Enter to begin another rollout. `Ctrl-C` is the emergency exit and intentionally skips automatic reset.

### Neutral-left proprio A/B test

For the right-arm-only model, this separate robot client replaces the six left-arm values sent to the model with the `right_arm_stack_bowl` training means. Live right-arm and both gripper values are still sent. Execution remains identical to the normal client, including the model's left-arm output, so the test changes only model conditioning.

Start the step-028200 server normally on the workstation, then run this command on the robot:

```bash
cd /home/r1lite/Documents/FastWAM

./experiments/r1lite/run_r1lite_client_neutral_left.sh \
  --server http://10.42.0.90:8000 \
  --prompt 'stack the bowl' \
  --policy-config right_arm_stack_bowl_neutral_left \
  --execute
```

The workstation server needs no modification. Remove `--execute` for a dry run. Use the normal client for the control run and this neutral-left client for the experimental run.

### Neutral-left convergence test

This experimental launcher keeps each action target active until all six right-arm joints are within 2 degrees for two fresh feedback samples, or until 250 ms has elapsed. The normal client is unchanged. Run it on the robot with:

```bash
cd /home/r1lite/Documents/FastWAM

./experiments/r1lite/run_r1lite_client_neutral_left_convergence.sh \
  --server http://10.42.0.90:8000 \
  --prompt 'stack the bowl' \
  --policy-config right_arm_stack_bowl_neutral_left_convergence \
  --execute
```

Override the defaults when needed with `--settle-tolerance-deg`, `--settle-timeout-s`, or `--settle-consecutive-samples`. Each action record includes whether it settled, its wait time, and its final maximum right-arm error.

## Observation and execution logs

At startup the robot client asks how many demos to run. The workstation server allocates the next numbered run and stores everything under:

```text
observations/run_N/
  run.json
  demo_M/
    events.jsonl
    action_steps.jsonl
    chunks/chunk_K/
      model_input.png
      model_input.pt
      request.json
      model_actions.json
      normalized_model_output.pt
```

`model_input.pt` contains the exact normalized image, proprioception, cached text context, mask, and inference arguments captured immediately before `infer_action`. `model_input.png` is the post-crop/post-resize three-camera mosaic before `[-1, 1]` normalization. Each action-step record compares the de-normalized model arm deltas and absolute gripper outputs with the measured state used for integration, the clipped absolute command, and ROS feedback measured after that control period. It also includes applied delta and tracking error fields.

Model input/output files and robot step uploads are written through background queues. Observation serialization and logging network calls do not block the inference or 15 Hz control loops. Use `--observations-dir` on the server to override the default `/home/spate308/Documents/FastWAM/observations` root.

## Replay one recorded demo with live-relative execution

`replay_hdf5_relative_robot.py` replays the stored actions from one selected HDF5 demo without using the model or workstation server. After the first Enter press, it interpolates both arms and grippers to that demo's recorded initial `joint_states[0]`, verifies the achieved pose, and then executes the demonstration using the same action interpretation as the FastWAM client:

```text
arm target[t] = live measured arm[t] + stored arm residual[t]
gripper target[t] = stored absolute gripper action[t]
```

Run a dry test on the robot first:

```bash
cd /home/r1lite/Documents/FastWAM

./experiments/r1lite/run_replay_hdf5_relative_robot.sh \
  --hdf5 /absolute/path/to/dataset.hdf5 \
  --demo demo_12 \
  --control-hz 20
```

After checking the selected demo, recorded initial state, and planned travel, enable movement:

```bash
./experiments/r1lite/run_replay_hdf5_relative_robot.sh \
  --hdf5 /absolute/path/to/dataset.hdf5 \
  --demo demo_12 \
  --control-hz 20 \
  --execute
```

Press Enter once to move to the selected demo's initial pose and begin replay. Press Enter again at any time to stop and hold the latest measured pose. Use `--max-steps N` for a short safety test. Logs are saved under `relative_hdf5_replay_results/run_<timestamp>/` on the robot.

### Offline replay of the seven longest recent live rollouts

The selection manifest and copied robot feedback logs are under
`offline_rollout_inputs/latest_15_longest_7/`. The selected demos are the seven
longest completed demos among the 15 most recent neutral-left trials. Stop the
live FastWAM server first so the offline process can load step 028200 on the GPU,
then run one to three chunks per demo:

```bash
cd /home/spate308/Documents/FastWAM
bash experiments/r1lite/run_offline_replay_live_rollouts.sh --chunks-per-demo 3
```

The script reruns the exact normalized image and proprio tensors saved immediately
before each live model call. For every executed action step, the offline and live
arm deltas are each added to the same recorded `measured_before` state. This
teacher-forced comparison separates:

- offline-versus-live model delta/target variation;
- live measured feedback versus the live commanded target (hardware tracking);
- live measured feedback versus the offline reconstructed target (combined error).

Results are written to `offline_rollout_results/`. Each demo gets delta and
target/feedback graphs, a compact `summary.json`, and full arrays in
`comparison.npz`; `aggregate_j2_j3_error.png` compares model rerun variation
with hardware tracking error across all seven demos. The live calls used
`seed=None`, so a rerun cannot reproduce the original diffusion noise. Pass
`--seed 0` only when a deterministic offline baseline is desired.

## Replay and debug a recorded demo

There are **two Python scripts**, each with a shell launcher:

- `replay_online_demo.py` runs on the **workstation**. It loads the model and dataset, stops at requested breakpoints, compares model inputs/actions with the recorded data, and **never moves the robot**.
- `replay_online_demo_robot.py` runs on the **robot**. It sends recorded inputs to the normal workstation server and can execute the returned actions using live ROS feedback. It does not load the model locally.

### 1. Inspect preprocessing and model output (workstation)

Use an HDF5 recording that corresponds to the selected checkpoint. The launcher defaults to the **right-arm-only `right_arm_stack_bowl`** step-001410 checkpoint, its config and stats, and the singular `stack the bowl` prompt. It does **not** require the HTTP server; stop the server first if GPU memory is tight. Never pair this checkpoint with the `stack_bowls_processed` config or stats.

```bash
cd /home/spate308/Documents/FastWAM
bash experiments/r1lite/run_replay_online_demo.sh \
  --hdf5 /absolute/path/to/right_arm_stack_bowl.hdf5 \
  --demo demo_0 --start-index 0 --num-chunks 1 \
  --break-at image --break-at state --break-at action --break-at control
```

At each `(Pdb)` prompt, use `p values` to inspect named arrays, `up` to inspect the calling code, and `c` to continue. `--break-at all` also stops before and after model loading/inference. The useful comparisons are:

- `image`: online model image versus the training-transform reference made from the same HDF5 frame; source, intermediate, and final sizes plus pixel error.
- `state`: raw recorded proprio versus normalized proprio, normalization scale/offset, and reconstruction error.
- `action`: raw and normalized model output versus raw and normalized ground-truth actions. Arms are deltas; grippers are absolute targets.
- `control`: predicted target versus target computed from ground-truth action, using recorded feedback at each step.

Results are saved under `replay_online_results/`: online/reference mosaic PNGs, a compact `summary.json`, and full arrays in `replay.npz`. This is teacher-forced offline replay, not a closed-loop simulation. To test a different `right_arm_stack_bowl` step with the same training setup, pass its path with `--checkpoint`.

### 2. Compare a saved live input with an HDF5 frame

After normal inference has saved a server chunk, find the newest exact model input:

```bash
cd /home/spate308/Documents/FastWAM
find observations -type f -name model_input.pt -printf '%T@ %h\n' | sort -nr | head
```

Choose a chunk directory from that output and an HDF5 frame representing approximately the same task stage:

```bash
bash experiments/r1lite/run_compare_live_hdf5_input.sh \
  --live-chunk observations/run_44/demo_1/chunks/chunk_10 \
  --hdf5 /home/spate308/Documents/stack_bowl_20260822_000619.hdf5 \
  --demo demo_0 --frame 0
```

The command compares raw and normalized proprio per dimension, lists live dimensions at the normalizer's `[-5, 5]` clamp, verifies image shapes, reports image error and inference settings, and writes `live_hdf5_comparison/live_left_hdf5_right.png`. Change `--demo` and `--frame` when comparing another stage. It uses the `right_arm_stack_bowl` config and statistics by default.

### 3. Optionally execute the replay on the robot

The robot must be near the recorded starting pose. The live client stops if any arm joint differs by more than 0.15 rad or either gripper differs by more than 15 units; `--max-arm-pose-error` and `--max-gripper-pose-error` control these checks. Recorded actions are used only as comparison data, never as commands.

First start the matching model server on the workstation:

```bash
cd /home/spate308/Documents/FastWAM
bash experiments/r1lite/run_fastwam_server.sh \
  --checkpoint runs/right_arm_stack_bowl/step_001410.pt \
  --dataset-stats runs/right_arm_stack_bowl/dataset_stats.json \
  --config configs/right_arm_stack_bowl_config.yaml \
  --text-embedding-cache-dir data/text_embeds_cache/r1_lite_tasks \
  --host 0.0.0.0 --port 8000
```

Copy the **matching** HDF5 recording to the robot; replace the example source path:

```bash
ssh r1lite@10.42.0.44 'mkdir -p ~/Documents/FastWAM/data_replay'
scp /absolute/path/to/right_arm_stack_bowl.hdf5 \
  r1lite@10.42.0.44:/home/r1lite/Documents/FastWAM/data_replay/
```

On the robot, run a one-chunk dry run first:

```bash
bash ~/Documents/FastWAM/experiments/r1lite/run_replay_online_demo_robot.sh \
  --hdf5 ~/Documents/FastWAM/data_replay/right_arm_stack_bowl.hdf5 \
  --demo demo_0 --server http://10.42.0.90:8000 \
  --prompt 'stack the bowl' --start-index 0 --num-chunks 1
```

Rerun that command with `--execute` only when ready to move the robot. Press Enter to start; press Enter again to stop and reset. The default execution rate is 20 Hz; `--replan-steps N` uses only the first `N` actions from each chunk. The robot writes a live command/feedback/ground-truth trace under `~/Documents/FastWAM/replay_online_results/`.

## Decisive demo 21–30 tests

These paired tests use the `right_arm_stack_bowl` step-028200 model and the
singular prompt `stack the bowl`. Offline evaluation covers each complete demo;
the physical robot test defaults to five 32-action chunks per demo.

### A. Offline GT versus predictions from seeds 0 and 1

Stop the inference server first so the offline model can use the GPUs, then run
on the workstation:

```bash
cd /home/spate308/Documents/FastWAM
bash experiments/r1lite/run_offline_demo_21_30.sh
```

This writes ten charts under `demo_21_30_offline_results/`, one for each of
`demo_21` through `demo_30`. Every chart overlays recorded ground truth with
predicted actions from seeds 0 and 1. Arms are relative deltas; grippers are
absolute targets. The final prediction is truncated when fewer than 32 recorded
steps remain. Charts show only right J1–J6 and the right gripper. Right-arm
curves and MAE are displayed in degrees; the gripper remains in native units.
Use `--num-chunks N`
to request a shorter offline test.

### B. Execute a precomputed offline trajectory on the robot

This hardware-isolation test does not use the workstation server. It loads the
saved offline seed trajectory and reconstructs each absolute arm target as
`recorded_state[t] + predicted_delta[t]`; grippers remain absolute. Enter first
starts a smooth move to the demo's recorded initial pose. The script then holds
for 1.2 seconds to simulate inference before every 32-step chunk and executes
targets at 20 Hz. Only the right arm and right gripper follow model outputs;
the left side is held at its live starting pose. Enter again stops and holds
immediately.

On the robot, perform a dry run first:

```bash
bash ~/Documents/FastWAM/experiments/r1lite/run_replay_demo_21_30_robot.sh \
  --seed 0
```

When ready for physical motion, add `--execute`. Use `--seed 1` for the second
offline trajectory. The launcher defaults to five chunks; add `--all-chunks`
to execute the complete saved trajectory. For every source demo, press Enter once to position/start
and again to stop. The client resets between demos. Results
go under `~/Documents/FastWAM/demo_21_30_robot_results/run_.../`; each of the
ten demo folders contains `model_target_vs_measured.png`, a tracking summary,
and full command/feedback records. Target and measured feedback are continuous
lines. Charts show right J1–J6 in degrees and the right gripper in native units,
with target-versus-measured MAE in every panel.
