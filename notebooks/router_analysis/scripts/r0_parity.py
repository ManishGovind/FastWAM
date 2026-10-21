"""R0 train/infer parity for the trained inline router.

Training feeds the router R0 = _encode_video_latents(full window)[:, :, 0].
Inference feeds R0 = _encode_input_image_latents_tensor(first frame only).
Same windows, same context: compare latents and router picks.
"""
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

REPO = "/data/mgovind/FastWAM"
RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_router_new/2026-10-06_21-44-40"
CKPT = f"{RUN}/checkpoints/weights/step_021700.pt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
os.chdir(REPO)
torch.manual_seed(0)

with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_new"])
model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
model.load_checkpoint(CKPT)
model = model.to("cuda").eval()
assert not model.modality_router.training
print("model loaded; router eval mode", flush=True)

dataset = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
idx = np.linspace(0, len(dataset) - 1, N).round().astype(int)
print(f"dataset len {len(dataset)}, testing {N} windows", flush=True)
names = model._enabled_streams()

rel_err, cos, agree_p = [], [], []
pick_train, pick_infer, pick_infer_ctx = [], [], []
dp = []
B = 4
with torch.no_grad():
    for s in range(0, N, B):
        batch = default_collate([dataset[int(i)] for i in idx[s:s + B]])
        b = batch["video_rgb"].shape[0]
        # --- training path (as in training_loss) ---
        lat, _, _ = model._encode_stream_latents(batch["video_rgb"], tiled=False)
        r0_train = lat[:, :, 0]
        shared = model._build_shared_context_action(sample=batch, batch_size=b, num_frames=int(batch["video_rgb"].shape[2]))
        ctx, ctx_mask = shared["context"], shared["context_mask"]
        lg_train = model.modality_router(model._build_router_features(r0_latent=r0_train, context=ctx, context_mask=ctx_mask))
        # --- inference path: single-frame encode of the same first frame ---
        r0_inf = torch.cat([
            model._encode_input_image_latents_tensor(
                input_image=batch["video_rgb"][j:j + 1, :, 0].to("cuda", torch.bfloat16))[:, :, 0]
            for j in range(b)
        ])
        lg_inf = model.modality_router(model._build_router_features(r0_latent=r0_inf, context=ctx, context_mask=ctx_mask))
        # --- inference context path too (cached text + proprio[0] via _resolve_infer_context) ---
        ctxs = [model._resolve_infer_context(prompt=None, proprio=batch["proprio"][j, 0],
                                             context=batch["context"][j], context_mask=batch["context_mask"][j])
                for j in range(b)]
        ictx = torch.cat([c for c, _ in ctxs]); imask = torch.cat([m for _, m in ctxs])
        lg_inf2 = model.modality_router(model._build_router_features(r0_latent=r0_inf, context=ictx, context_mask=imask))

        a, c_ = r0_train.float().flatten(1), r0_inf.float().flatten(1)
        rel_err += ((a - c_).norm(dim=1) / a.norm(dim=1)).tolist()
        cos += torch.nn.functional.cosine_similarity(a, c_, dim=1).tolist()
        pick_train += lg_train.argmax(-1).tolist(); pick_infer += lg_inf.argmax(-1).tolist(); pick_infer_ctx += lg_inf2.argmax(-1).tolist()
        dp += (lg_train.softmax(-1) - lg_inf.softmax(-1)).abs().max(-1).values.tolist()
        if s == 0:
            print(f"context identical (train vs infer path): {torch.allclose(ctx.float(), ictx.float(), atol=1e-3)}", flush=True)

pt, pi, pc = map(np.array, (pick_train, pick_infer, pick_infer_ctx))
print(f"\nR0 latent  ||train - infer|| / ||train||: mean {np.mean(rel_err):.4f}  max {np.max(rel_err):.4f}")
print(f"R0 latent  cosine similarity:            mean {np.mean(cos):.4f}  min {np.min(cos):.4f}")
print(f"router argmax agreement train-R0 vs infer-R0: {np.mean(pt == pi):.3f}  (with infer context too: {np.mean(pt == pc):.3f})")
print(f"max |prob diff| per window: mean {np.mean(dp):.3f}  max {np.max(dp):.3f}")
for nm, arr in (("train-path picks", pt), ("infer-path picks", pi), ("infer-path + infer ctx", pc)):
    cnt = np.bincount(arr, minlength=len(names))
    print(f"{nm:24s} " + "  ".join(f"{n} {c / len(arr):.2f}" for n, c in zip(names, cnt)))
