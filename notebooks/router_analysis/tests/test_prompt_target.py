"""modality_prompt_target: which text context each expert receives (training + inference).

Runs the real model on CPU (fp32) so it never competes with a training job for GPU memory.
Run from the repo root: python notebooks/router_analysis/tests/test_prompt_target.py
"""
import hashlib, os
import numpy as np, torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

REPO = "/data/mgovind/FastWAM"; RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_router_new/2026-10-06_21-44-40"
os.chdir(REPO); torch.manual_seed(0)
with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_short_m_context"])
model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
model = model.eval(); model.modality_router.train()
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
batch = default_collate([ds[int(i)] for i in np.linspace(0, len(ds) - 1, 2).round().astype(int)])

seen = {"video": [], "action": []}
vp, ap = model.video_expert.prepare, model.action_expert.prepare
def spy_v(*a, **k): seen["video"].append(k["context"].detach().clone()); return vp(*a, **k)
def spy_a(*a, **k): seen["action"].append(k["context"].detach().clone()); return ap(*a, **k)
model.video_expert.prepare, model.action_expert.prepare = spy_v, spy_a

def first_tokens(ctx):  # instruction part only (drop proprio token)
    return ctx[..., :128, :].float()

plain = batch["context"].float()  # dataset context = T5(plain instruction), no proprio
for target in ("both", "video"):
    model.modality_prompt_target = target
    for k in seen: seen[k].clear()
    with torch.no_grad():
        _, d = model.training_loss(batch)
    def is_plain(ctxs):  # every recorded row equals the plain instruction of some sample
        rows = torch.cat([first_tokens(c) for c in ctxs])
        return all(min(float((r - q).abs().max()) for q in plain) < 1e-3 for r in rows)
    vs_plain_v, vs_plain_a = is_plain(seen["video"]), is_plain(seen["action"])
    print(f"[train] target={target:5s} | video expert gets plain instruction: {vs_plain_v} | "
          f"action expert gets plain instruction: {vs_plain_a} | loss_action {d['loss_action']:.5f}", flush=True)

# inference (eval path: cached context + prompt_text), 2 denoising steps
s0 = ds[0]
for target in ("both", "video"):
    model.modality_prompt_target = target
    for k in seen: seen[k].clear()
    model.eval()
    with torch.no_grad():
        out = model.infer_action(prompt=None, input_image=s0["video_rgb"][:, 0], action_horizon=int(s0["action"].shape[0]),
                                 num_video_frames=int(s0["video_rgb"].shape[1]), proprio=s0["proprio"][0],
                                 context=s0["context"], context_mask=s0["context_mask"], num_inference_steps=2, seed=0,
                                 prompt_text=s0["prompt"])
    v, a = seen["video"][-1], seen["action"][-1]
    p = s0["context"].float()[None]
    print(f"[infer] target={target:5s} | chosen {out['active_future']} | video expert plain: "
          f"{torch.allclose(first_tokens(v), p, atol=1e-3)} | action expert plain: {torch.allclose(first_tokens(a), p, atol=1e-3)}", flush=True)
