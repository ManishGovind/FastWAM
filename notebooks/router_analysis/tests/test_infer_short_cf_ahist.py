"""cf router + action history (step 21700): full vs short inference layout.

full  = [R0 | R_fut | D_fut | F_fut | A], only the chosen future denoised, action gated to it (as trained)
short = [R0 | chosen future | A]
Same seed, same action history -> the router pick and the actions should match (up to bf16 noise),
then a short-layout LIBERO eval reproduces the full-layout one at single-stream cost.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/tests/test_infer_short_cf_ahist.py [fp32]
"""
import hashlib, os, sys, time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

DEV = "cuda"
REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_tok_embed_cf_ahist"
RUN = f"{REPO}/runs/{TASK}/2026-11-01_00-27-41"
DT = torch.float32 if sys.argv[1:] == ["fp32"] else torch.bfloat16
os.chdir(REPO)
with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=[f"task={TASK}"])
model = instantiate(cfg.model, model_dtype=DT, device=DEV)
model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
model = model.to(DEV).eval()
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")


def cached_encode(prompt):  # cached T5 (no text encoder loaded)
    out = []
    for t in ([prompt] if isinstance(prompt, str) else list(prompt)):
        h = hashlib.sha256(t.encode()).hexdigest()
        for d in ("data/text_embeds_cache/libero_modality", "data/text_embeds_cache/libero"):
            f = f"{d}/{h}.t5_len128.wan22ti2v5b.pt"
            if os.path.exists(f):
                out.append(torch.load(f))
                break
        else:
            raise FileNotFoundError(t)
    return torch.stack([o["context"] for o in out]).to(DEV).to(DT), torch.stack([o["mask"] for o in out]).to(DEV)


model.text_encoder = object()
model.encode_prompt = cached_encode
seq = []
orig = model._build_multistream_mot_attention_mask
model._build_multistream_mot_attention_mask = lambda *a, **k: seq.append(tuple((m := orig(*a, **k)).shape)) or m

print("dtype", DT, flush=True)
worst = 0.0
for i in np.linspace(0, len(ds) - 1, 6).round().astype(int):
    s = ds[int(i)]
    res = {}
    for layout in ("full", "short"):
        model.router_layout = layout
        seq.clear()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            out = model.infer_action(
                prompt=s["prompt"], input_image=s["video_rgb"][:, 0].to(DEV).to(DT),
                action_horizon=int(s["action"].shape[0]), num_video_frames=int(s["video_rgb"].shape[1]),
                proprio=s["proprio"][0].to(DEV), num_inference_steps=10, seed=0,
                action_history=s["action_history"][None].to(DEV), action_history_is_pad=s["action_history_is_pad"][None].to(DEV),
            )
        torch.cuda.synchronize()
        res[layout] = (out["action"], out["active_future"], time.time() - t0, seq[0])
    a, b = res["full"][0], res["short"][0]
    rel = float((a - b).abs().max() / a.abs().max())
    worst = max(worst, rel)
    print(f"window {int(i):6d}: pick full={res['full'][1]} short={res['short'][1]} | mask full {res['full'][3]} short {res['short'][3]} | "
          f"rel max|action diff| {rel:.2e} | time full {res['full'][2]:.2f}s short {res['short'][2]:.2f}s", flush=True)
    assert res["full"][1] == res["short"][1], "router pick differs between layouts"
print(f"worst rel diff {worst:.2e}", "OK" if worst < 2e-2 else "MISMATCH")
