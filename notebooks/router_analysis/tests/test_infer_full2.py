import hashlib, os, sys, time
import numpy as np, torch
DEV = __import__("os").environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # TEST_DEVICE=cpu to avoid GPUs
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate
REPO = "/data/mgovind/FastWAM"; RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_router_new/2026-10-06_21-44-40"
os.chdir(REPO)
DT = torch.float32 if sys.argv[1:] == ["fp32"] else torch.bfloat16
with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_new_full_prompt"])
model = instantiate(cfg.model, model_dtype=DT, device=DEV)
model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
model = model.to(DEV).eval()
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
def fake_encode(prompt):
    texts = [prompt] if isinstance(prompt, str) else list(prompt); out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).hexdigest()
        for d in ("data/text_embeds_cache/libero_modality", "data/text_embeds_cache/libero"):
            f = f"{d}/{h}.t5_len128.wan22ti2v5b.pt"
            if os.path.exists(f): out.append(torch.load(f)); break
        else: raise FileNotFoundError(t)
    return torch.stack([o["context"] for o in out]).to(DEV).to(DT), torch.stack([o["mask"] for o in out]).to(DEV)
model.text_encoder = object(); model.encode_prompt = fake_encode
seqlens = []
orig = model._build_multistream_mot_attention_mask
def spy(*a, **k):
    m = orig(*a, **k); seqlens.append((tuple(m.shape), str(m.dtype))); return m
model._build_multistream_mot_attention_mask = spy
loop_info = []
orig_loop = model._run_multistream_infer_loop
def loop_spy(**k):
    init = {n: t.clone() for n, t in k["latents_video"].items()}
    lv, la = orig_loop(**k)
    changed = {n: bool((lv[n][:, :, 1:] - init[n][:, :, 1:]).abs().max() > 0) for n in init}
    loop_info.append((k.get("update_streams"), changed))
    return lv, la
model._run_multistream_infer_loop = loop_spy
print("dtype", DT, "layout in config:", model.router_layout, flush=True)
for i in np.linspace(0, len(ds) - 1, 3).round().astype(int):
    s = ds[int(i)]; res = {}
    for layout in ("full", "short"):
        model.router_layout = layout; seqlens.clear(); loop_info.clear()
        (torch.cuda.synchronize() if DEV == 'cuda' else None); t0 = time.time()
        with torch.no_grad():
            out = model.infer_action(prompt=s["prompt"], input_image=s["video_rgb"][:, 0].to(DEV).to(DT),
                                     action_horizon=int(s["action"].shape[0]), num_video_frames=int(s["video_rgb"].shape[1]),
                                     proprio=s["proprio"][0].to(DEV), num_inference_steps=10, seed=0)
        (torch.cuda.synchronize() if DEV == 'cuda' else None)
        res[layout] = (out["action"], out["active_future"], time.time() - t0, seqlens[0], loop_info[0])
    a, b = res["full"][0], res["short"][0]
    print(f"window {int(i):6d}: router -> {res['full'][1]} | MoT mask full {res['full'][3]} vs short {res['short'][3]} | "
          f"full: denoised {res['full'][4][0]} -> futures changed {res['full'][4][1]} | "
          f"max|action diff| {float((a - b).abs().max()):.2e} (|action| ~{float(a.abs().mean()):.3f}) | time full {res['full'][2]:.1f}s short {res['short'][2]:.1f}s", flush=True)
