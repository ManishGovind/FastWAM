# R1 Lite: inference, dashboard, and HDF5 overlay

Run the **policy server on the workstation first**. Then run one robot client
in a separate terminal. The examples below execute actions; omit `--execute`
for a dry run.

Replace these placeholders before running a client:

- `<WORKSTATION_WIRED_IP>`: workstation IP reachable from the robot.
- `<PROMPT>`: task instruction used by the model.
- `<POLICY_CONFIG>`: policy/run label. It becomes the rollout HDF5 filename
  and dashboard grouping name.

## Galaxea robot topics (robot)

Start the R1-Lite hardware/session before any dashboard, camera overlay, or
policy-client command. Run this in its own robot terminal and leave it running:

```bash
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d
```

Verify that ROS topics are available from a second terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/galaxea/install/setup.bash
ros2 topic list
```

When all evaluation clients and tools have stopped, stop the Galaxea session
with:

```bash
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh kill
```

## Live evaluation flow

Run the steps in this order for either policy family:

```text
1. Robot: start the Galaxea R1-Lite topic session above.
2. Workstation: start the policy server.
3. Robot: start the dashboard (optional, but recommended).
4. Robot: start one policy client with --execute.
5. Per trial: Enter to start, Enter to stop/reset, then assign a 0–100 score.
6. Review the HDF5 rollout and score in the dashboard or with the overlay.
```

Use one `--policy-config` label for a complete evaluation set. Reusing that
label resumes its newest incomplete HDF5 run; use `--no-resume` to begin a
separate set intentionally. Do not run FastWAM and OpenPI clients at the same
time: both command the same robot.

## FastWAM policy server (workstation)(Folder under spate308 but skapse has full access to the workspace folder which contains both FastWAM & OpenPI to start the model servers)

Start this before the FastWAM robot client. Substitute the checkpoint, stats,
config, and embedding-cache directory for the task being evaluated.
You can simply go up in terminal where you will see the command for the latest model tested.

```bash
cd /home/spate308/workspace/FastWAM

bash experiments/r1lite/run_fastwam_server.sh \
  --checkpoint runs/<RUN_NAME>/<CHECKPOINT>.pt \
  --dataset-stats runs/<RUN_NAME>/dataset_stats.json \
  --config configs/<TASK_CONFIG>.yaml \
  --text-embedding-cache-dir data/text_embeds_cache/<CACHE_DIR> \
  --host 0.0.0.0 \
  --port 8000
```

The workstation terminal should finish with `Uvicorn running on
http://0.0.0.0:8000`. Leave that process running while evaluating. Stop it
with `Ctrl+C` only after the robot client has stopped.

### FastWAM inference implementation map

This is the exact live FastWAM call path. The same source files exist under
`/home/r1lite/Documents/FastWAM/experiments/r1lite/` on the robot and
`/home/spate308/workspace/FastWAM/experiments/r1lite/` on the workstation.

```text
Robot launch script
  run_r1lite_client.sh
    -> inference_r1lite_fastwam.py:main()
       -> RobotIO.synchronized_observation()
       -> RobotIO.make_payload()
       -> request_actions_while_spinning()
          -> HTTP POST <server>/infer

Workstation launch script
  run_fastwam_server.sh
    -> serve_fastwam_r1lite.py:main()
       -> FastWAMR1LitePolicy.infer(request)
          -> self.model.infer_action(...)
```

`FastWAMR1LitePolicy.__init__()` builds the configured model with Hydra,
loads the selected checkpoint, and loads the dataset normalizer. In
`FastWAMR1LitePolicy.infer()`, the server decodes the three images, composes
the FastWAM image tensor, loads the cached text embedding, normalizes the
14-D robot state, and invokes `self.model.infer_action(...)`. It then
denormalizes and clips the returned `[action_horizon, 14]` action chunk to the
dataset action range before replying to the client.

For the current absolute-action checkpoints, the robot client executes each
returned action with:

```text
RobotIO.prepare_absolute_targets(action, gripper_min, gripper_max)
RobotIO.publish_targets_if_enabled(target, ...)
```

The client holds the latest target between policy chunks with
`RobotIO.start_hold()`. Relative-mode checkpoints instead use
`RobotIO.prepare_targets(...)`, which adds the predicted arm delta to live
joint feedback.

### FastWAM evaluation sequence

```bash
# Terminal 1 — workstation: start the FastWAM server above.

# Terminal 2 — robot: optional dashboard.
cd /home/r1lite/Documents
bash start_dashboard.sh

# Terminal 3 — robot: execute trials.
cd /home/r1lite/Documents/FastWAM
bash experiments/r1lite/run_r1lite_client.sh \
  --server http://<WORKSTATION_WIRED_IP>:8000 \
  --prompt "<PROMPT>" \
  --policy-config "<POLICY_CONFIG>" \
  --action-mode absolute \
  --control-hz 20 \
  --execute
```

FastWAM rollouts are written to:

```text
/home/r1lite/Documents/FastWAM/rollouts/r1lite/<POLICY_CONFIG>/<DATE>/<RUN>/<POLICY_CONFIG>.hdf5
```

Open the dashboard at `http://10.42.0.44:8080` to review trial scores and
rename, score, or remove saved trials/runs. The overlay command at the end of
this guide can inspect the saved agent-view images frame by frame.

## Dashboard (robot)

```bash
cd /home/r1lite/Documents

# Builds if needed, then serves the dashboard on port 8080.
bash start_dashboard.sh
```

Open `http://10.42.0.44:8080` from a browser. If the frontend was already
built and only the backend needs restarting:

```bash
bash start_dashboard.sh --skip-build
```

## FastWAM client (robot)

Start the FastWAM server on the workstation before this command. The current
absolute-action FastWAM models must use `--action-mode absolute`.

```bash
cd /home/r1lite/Documents/FastWAM

bash experiments/r1lite/run_r1lite_client.sh \
  --server http://<WORKSTATION_WIRED_IP>:8000 \
  --prompt "<PROMPT>" \
  --policy-config "<POLICY_CONFIG>" \
  --action-mode absolute \
  --control-hz 20 \
  --execute
```

The client asks for the number of trials if the run is not resumed. Press
Enter to start each trial, Enter again to stop/reset, then enter a score from
`0` to `100`. HDF5 rollout files are saved robot-local under:

```text
/home/r1lite/Documents/FastWAM/rollouts/r1lite/
```

Useful FastWAM options:

```bash
# Start a new run instead of resuming an unfinished run with the same label.
--no-resume

# Execute only this many predicted actions before requesting the next chunk.
--replan-steps 8
```


## HDF5 live-camera overlay (robot)

This overlays one saved agent-view frame on the live head camera and serves a
small viewer at `http://10.42.0.44:8765`.

```bash
cd /home/r1lite/Documents

OVERLAY_HDF5="/absolute/path/to/rollout_or_dataset.hdf5" \
OVERLAY_DEMO=0 \
OVERLAY_PORT=8765 \
bash run_hdf5_overlay.sh
```

Optional overlay controls:

```bash
# Start at a particular frame and choose a non-default camera topic.
bash run_hdf5_overlay.sh --frame 25 \
  --topic /hdas/camera_head/left_raw/image_raw_color/compressed

# The browser UI can select demo/frame, opacity, and blend/outline/side-by-side mode.
```

Use `Ctrl+C` in the corresponding terminal to stop a client, overlay, or
foreground dashboard process.



## OpenPI client (robot)

Start the OpenPI websocket server on the workstation before this command.
The OpenPI client now executes policy output as **absolute targets by default**;
`--action-mode absolute` is included explicitly for clarity.

```bash
cd /home/r1lite/Documents/openpi-r1lite

bash examples/r1lite/run_inference_client.sh \
  --host <WORKSTATION_WIRED_IP> \
  --port 8000 \
  --prompt "<PROMPT>" \
  --policy-config "<POLICY_CONFIG>" \
  --action-mode absolute \
  --num-demos 20 \
  --execute
```

Press Enter to start a trial and Enter again to stop/reset. After reset,
enter the score from `0` to `100`. This saves the existing rollout video and
command-feedback JSON plus a dashboard-compatible HDF5 rollout under:

```text
/home/r1lite/Documents/openpi-r1lite/rollouts/r1lite/<POLICY_CONFIG>/<DATE>/<RUN>/<POLICY_CONFIG>.hdf5
```

OpenPI automatically resumes the newest incomplete HDF5 run for the same
`--policy-config`, preserving its completed trials and continuing at the next
trial number. To force a separate run instead, add:

```bash
--no-resume
```

Only use this older convention with an old residual-action OpenPI checkpoint:

```bash
--action-mode relative
```

## OpenPI multi-task-5 server (workstation)

Run this on the workstation for the `30000` checkpoint:

```bash
cd /home/spate308/workspace/openpi-r1lite

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_r1lite_joint_5task_v1 \
  --policy.dir /home/spate308/workspace/openpi-r1lite/checkpoints/baseline_checkpoints/multi-task-5/30000
```
