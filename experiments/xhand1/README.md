# xArm7 + XHand1 FastWAM inference

Adapted from `experiments/r1lite` (branch `real-world-mm`).

- `serve_fastwam_xhand1.py` runs on the GPU host. It owns all FastWAM work:
  config + checkpoint loading, the training image path (processor
  `val_transforms` per camera, then the configured camera layout), state normalization,
  inference, de-normalization, and clipping to the dataset action range.
- `inference_xhand1_fastwam.py` runs on the robot workstation, on top of
  `xhand1_hardware.py`: the hardware layer of Dev_dex_hand_tactile
  `vpp_real/scripts/deploy_policy.py` vendored verbatim except lines marked
  `FastWAM:` (Open-Teach camera streams and ROI crops in `camera_rois.json`,
  `SafeRobotExecutor` with workspace / IK / per-step / XHand limits and shadow
  mode by default, XHand startup reset, HDF5 rollout recording, terminal
  scoring). No vpp_real checkout or torch is needed; Open-Teach still provides
  the XHand driver, `clamp_pose_step`, and the camera publisher.

## Contract

```text
state:   [x, y, z (m), rx, ry, rz (axis-angle rad), 12 measured XHand joints (rad)]
images:  post-ROI 480x480 JPEGs keyed by the checkpoint's cameras, either
           xhand1_tasks_joint_3cam_384_1e-4: exo above [ego | wrist], 384x320 (robotwin)
           xhand1_tasks_joint_2cam224_1e-4:  ego | wrist, 224x448 (LIBERO 2-cam)
actions: (32, 18), same layout as state but commanded hand joints
cadence: policy_hz = 30 Hz / global_sample_stride (stride 1: 32 steps = 1.07 s; stride 3: 3.2 s)
```

Action semantics follow r1lite: the server reports `EVALUATION.action_semantics`
(default `absolute` here, since the xhand1 data is absolute) and the client's
`--action-mode` must match. A relative checkpoint marks its delta dims with
`processor.delta_action_dim_mask`; the client accumulates those per-step
increments onto the measured state (rotation composed in the tool frame) and
passes the other dims through as absolute targets.

The client executes the first `--execute-steps` targets (default 8: 0.8 s at
stride 3, like a VPP chunk; 0.27 s at stride 1), interpolated to the 30 Hz
executor, then re-plans. It
checks `/metadata` before opening any SDK and refuses to start on a dimension,
camera, absolute/relative, or cadence mismatch.

## One-time preparation (GPU host, FastWAM environment)

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
# Prompts come from the configured datasets' meta/tasks.jsonl; both variants share
# the datasets and data/text_embeds_cache/xhand1_tasks, so either task works
python scripts/precompute_text_embeds.py task=xhand1_tasks_joint_3cam_384_1e-4
```

The prompt passed to the client must exactly match one of the task strings:
`stack the cups`, `flip the notebook`, `clean the whiteboard`.

## Run

GPU host. The server binds to 127.0.0.1 by default (it has no authentication),
so the robot workstation reaches it through an SSH tunnel, like the VPP
inference server; pass `--host 0.0.0.0` only on a trusted network:

```bash
bash experiments/xhand1/run_fastwam_server.sh \
  --config runs/<RUN>/config.yaml \
  --checkpoint runs/<RUN>/checkpoints/weights/<STEP>.pt \
  --dataset-stats runs/<RUN>/dataset_stats.json \
  --port 8000
```

`--config` is the config snapshot training writes next to the checkpoint, so
later edits under `configs/` or command-line overrides used for that run cannot
diverge from the weights. `--config configs/task/<task>.yaml` (either variant)
also works (composed as `train.yaml` + `task=...`, the same way training does)
when the configs are known to be unchanged since training. The client takes
its camera list from the server, so the same client command serves both.

Robot workstation, in a Python env with numpy, opencv-python, scipy, pyzmq,
h5py, `xarm-python-sdk`, and Open-Teach's XHand dependencies
(`ssh -N -L 8000:127.0.0.1:8000 <gpu-host>` first):

```bash
python experiments/xhand1/inference_xhand1_fastwam.py \
  --openteach-root /path/to/Open-Teach \
  --server http://127.0.0.1:8000 --prompt "stack the cups" \
  --record-h5 rollouts/fastwam_stack_cups.hdf5 \
  --execution shadow            # then controller-sim, then physical (type MOVE)
```

Server-side model inputs and outputs are saved per chunk under
`server_observations/run_N/demo_1/chunks/chunk_K/`.
