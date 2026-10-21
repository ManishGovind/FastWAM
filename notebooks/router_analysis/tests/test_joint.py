"""Real-model checks for modality_prompt + router_layout in FastWAMJointMultimodalSelectNew."""
import os
import sys
DT = torch_dtype = None

import numpy as np
import torch
DEV = __import__("os").environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # TEST_DEVICE=cpu to avoid GPUs
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

REPO = "/data/mgovind/FastWAM"
RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_router_new/2026-10-06_21-44-40"
os.chdir(REPO)
with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_new", "model.modality_prompt=true"])
DT = torch.float32 if (len(sys.argv) > 1 and sys.argv[1] == "fp32") else torch.bfloat16
print("dtype", DT)
model = instantiate(cfg.model, model_dtype=DT, device=DEV)
model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
model = model.to(DEV).eval()
model.modality_router.train()  # sample choices like in training
print("mode", model.modality_prompt_mode, "sentences", model.modality_prompt_sentences)
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
batch = default_collate([ds[int(i)] for i in np.linspace(0, len(ds) - 1, 4).round().astype(int)])

# joint context check: [T5(instruction + condition) | proprio]
ctx, msk = batch["context"].to(DEV).to(DT), batch["context_mask"].to(DEV)
prop = model.proprio_encoder(batch["proprio"][:, 0].to(DEV).to(DT).unsqueeze(1))
c0 = torch.cat([ctx, prop], 1); m0 = torch.cat([msk, torch.ones(4, 1, dtype=torch.bool, device=DEV)], 1)
names = ["rgb", "depth", "flow", "depth"]
c1, m1 = model._modality_prompt_context(c0, m0, names, prompts=batch["prompt"])
ref = [model._load_joint_prompt(model._joint_prompt_text(p, n))[0] for p, n in zip(batch["prompt"], names)]
ok = all(torch.allclose(c1[b, :128].float().cpu(), ref[b].float(), atol=1e-2) and torch.equal(c1[b, -1], c0[b, -1]) for b in range(4))
print(f"joint context: {tuple(c0.shape)} -> {tuple(c1.shape)}; = cached combined + same proprio: {ok}; "
      f"differs from plain instruction: {bool((c1[:, :128] - c0[:, :128]).abs().amax() > 1e-3)}")
print("example text:", repr(model._joint_prompt_text(batch["prompt"][1], "depth")))

def run(name, *, prompt, layout, vloss, seed=0):
    model.modality_prompt, model.router_layout, model.router_video_loss = prompt, layout, vloss
    torch.manual_seed(seed); (torch.cuda.manual_seed(seed) if DEV == 'cuda' else None)
    with torch.no_grad():
        _, d = model.training_loss(batch)
    picks = [round(d[f"router_pick_{s}"], 2) for s in ("rgb", "depth", "flow")]
    print(f"{name:34s} loss_action {d['loss_action']:.6f}  loss_video {d['loss_video']:.6f}  "
          f"v_rgb/depth/flow {d['loss_video_rgb']:.4f}/{d['loss_video_depth']:.4f}/{d['loss_video_flow']:.4f}  picks {picks}")
    return d


a = run("A short, no prompt", prompt=False, layout="short", vloss="chosen")
b = run("B short, prompt", prompt=True, layout="short", vloss="chosen")
c = run("C full, prompt, video loss chosen", prompt=True, layout="full", vloss="chosen")
d = run("D full, prompt, video loss all", prompt=True, layout="full", vloss="all")
print(f"\nB vs A (prompt changes the result):        loss_action diff {abs(b['loss_action'] - a['loss_action']):.2e}")
print(f"C vs B (full == short for the chosen path): loss_action diff {abs(c['loss_action'] - b['loss_action']):.2e}, "
      f"loss_video diff {abs(c['loss_video'] - b['loss_video']):.2e}")
print(f"D vs C (all adds the unchosen futures):     loss_video {c['loss_video']:.4f} -> {d['loss_video']:.4f}, "
      f"loss_action diff {abs(d['loss_action'] - c['loss_action']):.2e}")

if DT == torch.float32:
    sys.exit(0)
# inference with prompt (router picks; sentence appended)
model.eval(); model.modality_prompt = True
import hashlib
seen = []
def fake_encode(prompt):
    texts = [prompt] if isinstance(prompt, str) else list(prompt)
    out = []
    for t in texts:
        seen.append(t)
        h = hashlib.sha256(t.encode()).hexdigest()
        for d in ("data/text_embeds_cache/libero_modality", "data/text_embeds_cache/libero"):
            f = f"{d}/{h}.t5_len128.wan22ti2v5b.pt"
            if os.path.exists(f):
                out.append(torch.load(f)); break
        else:
            raise FileNotFoundError(t)
    return (torch.stack([o["context"] for o in out]).to(DEV).to(DT), torch.stack([o["mask"] for o in out]).to(DEV))
model.text_encoder = object()       # pretend T5 is loaded (eval config)
model.encode_prompt = fake_encode   # ... and serve it from the caches
img = batch["video_rgb"][0, :, 0].to(DEV).to(DT)
with torch.no_grad():
    out = model.infer_action(prompt=batch["prompt"][0], input_image=img, action_horizon=int(batch["action"].shape[1]),
                             num_video_frames=int(batch["video_rgb"].shape[2]), proprio=batch["proprio"][0, 0].to(DEV),
                             num_inference_steps=4, seed=0)
print("infer_action (joint): active_future", out["active_future"], "| texts encoded:", [t[-60:] for t in seen],
      "| action", tuple(out["action"].shape), "finite", bool(torch.isfinite(out["action"]).all()))
