"""R1-Lite cf router + action history config (r1_lite_..._embed_tok_cf_ahist): train + eval paths, GPU, real data.

Weights: pretrained init, exactly what training starts from (the old R1 router ckpt has no act_proj).
Dataset stats: the R1 router_full_all_streamemb_embed run.
  1. data: action_history [32, 14] + is_pad [32]; 3 cams tiled to 384x320 per stream.
  2. training_loss (train mode, past warm-up, B=2): finite, counterfactual logs, router (incl. act_proj) gets grad.
  3. Trainer.evaluate path (eval mode, B=1, _to_batched_eval_sample): training_loss + infer run.
  4. infer_action with an action history (deploy path): router picks, action [32, 14].

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/tests/test_r1_cf_ahist.py
"""
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "r1_lite_joint_3cam_384_1e-4_router_full_all_streamemb_embed_tok_cf_ahist"
RUN = f"{REPO}/runs/r1_lite_joint_3cam_384_1e-4_router_full_all_streamemb_embed"
DEV, DT = "cuda", torch.bfloat16

os.chdir(REPO)
run = sorted(d for d in os.listdir(RUN))[-1]
RUN = f"{RUN}/{run}"
ckpt = sorted(os.listdir(f"{RUN}/checkpoints/weights"))[-1]
with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=[f"task={TASK}"])
model = instantiate(cfg.model, model_dtype=DT, device=DEV)
model = model.to(DEV)
assert model.router_reward == "counterfactual" and model.router_action_history
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
print(f"stats {run}; {len(ds)} windows", flush=True)

# 1. data
s = ds[len(ds) // 2]
assert tuple(s["action_history"].shape) == (32, 14) and tuple(s["action_history_is_pad"].shape) == (32,)
assert tuple(s["video_rgb"].shape[-2:]) == (384, 320), s["video_rgb"].shape
print(f"1. action_history {tuple(s['action_history'].shape)}, video_rgb {tuple(s['video_rgb'].shape)}  OK", flush=True)

# 2. training step past warm-up
Wan22Trainer._apply_dit_only_train_mode(model)
model.router_warmup_steps = 0
batch = default_collate([ds[0], ds[len(ds) // 3], ds[2 * len(ds) // 3]][:2])
model.zero_grad(set_to_none=True)
loss, logs = model.training_loss(batch)
loss.backward()
r = model.modality_router
g_act = sum(p.grad.float().abs().sum().item() for p in r.act_proj.parameters() if p.grad is not None)
assert torch.isfinite(loss) and "router_cf" in logs and g_act > 0
print(f"2. train loss {loss.item():.4f}; router_cf {logs['router_cf']:+.4f}; act_proj grad {g_act:.3e}  OK", flush=True)
model.zero_grad(set_to_none=True)

# 3. Trainer.evaluate path
model.eval()
smp = Wan22Trainer._to_batched_eval_sample(ds[len(ds) - 1])
with torch.no_grad(), torch.autocast("cuda", dtype=DT):
    vl, _ = model.training_loss(smp)
    pred = model.infer(
        prompt=None, input_image=smp["video"][0][:, 0].unsqueeze(0), num_frames=int(smp["video"].shape[2]),
        action=smp["action"][0], action_horizon=smp["action_horizon"], proprio=smp["proprio"][0, 0],
        context=smp["context"][0], context_mask=smp["context_mask"][0], text_cfg_scale=1.0, action_cfg_scale=1.0,
        num_inference_steps=4, seed=42, tiled=False, prompt_text=smp["prompt"][0],
    )
assert torch.isfinite(vl) and pred["action"] is not None
print(f"3. evaluate path (B=1): val loss {vl.item():.4f}, infer action {tuple(pred['action'].shape)}  OK", flush=True)

# 4. deploy path with history
with torch.no_grad():
    out = model.infer_action(
        prompt=None, input_image=s["video_rgb"][:, 0].to(DEV).to(DT), action_horizon=int(s["action"].shape[0]),
        num_video_frames=int(s["video_rgb"].shape[1]), proprio=s["proprio"][0].to(DEV),
        context=s["context"].to(DEV), context_mask=s["context_mask"].to(DEV), prompt_text=s["prompt"],
        num_inference_steps=4, seed=0,
        action_history=s["action_history"][None].to(DEV), action_history_is_pad=s["action_history_is_pad"][None].to(DEV),
    )
assert tuple(out["action"].shape) == (32, 14)
print(f"4. infer_action with history: pick {out['active_future']}, action {tuple(out['action'].shape)}  OK", flush=True)
print("ALL OK")
