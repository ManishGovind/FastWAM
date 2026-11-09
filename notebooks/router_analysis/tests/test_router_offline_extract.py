"""scripts/router_offline_extract.py + model._last_cf_cost (GPU, real cf+ahist weights + data).

  1. extraction (16 samples of shard 1/4, B=8) writes one chunk: features + costs [16, 3];
  2. the saved costs are the model's cf costs: re-running a batch with the same seed gives the same
     [B, 3], and its column means equal the logged loss_action_{rgb,depth,flow};
  3. router(cached features) == router(the model's full live features);
  4. the added line does not change training: router-only training step (only the router gets grads)
     and the Trainer.evaluate path (eval mode, B=1, scalar timestep weight) both run and record [B, 3].

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/tests/test_router_offline_extract.py
"""
import shutil
import sys

import torch
from torch.utils.data import default_collate

sys.path.insert(0, "scripts")
import router_offline_extract as ex  # noqa: E402
from fastwam.trainer import Wan22Trainer  # noqa: E402

OUT = "/data/mgovind/claude_scratch/router_offline_test"
shutil.rmtree(OUT, ignore_errors=True)
sys.argv = ["x", "--out", OUT, "--limit", "16", "--batch-size", "8", "--chunk", "16", "--shard", "1",
            "--num-shards", "4", "--num-workers", "4"]
ex.main()
import gc  # noqa: E402
gc.collect()
torch.cuda.empty_cache()  # free the extraction's model before loading the live one

# 1. chunk contents
d = torch.load(f"{OUT}/shard01_chunk00000.pt", weights_only=False)
n = 16
assert d["r0_latent"].shape == (n, 48, 14, 28), d["r0_latent"].shape
assert d["text"].shape == (n, 3072) and d["proprio"].shape == (n, 3072)
assert d["act_hist"].shape == (n, 256) and d["costs"].shape == (n, 3), d["costs"].shape
assert d["index"].tolist() == list(range(1, 64, 4))
assert torch.isfinite(d["costs"]).all() and (d["costs"] >= 0).all()
print("1. chunk:", {k: tuple(v.shape) for k, v in d.items() if torch.is_tensor(v)}, " OK", flush=True)

# 2./3. same model, first batch of the chunk again
sys.argv = ["x"]
model, ds = ex.load(ex.parse())
router, pe = model.modality_router, model.video_expert.patch_embedding
orig = type(model)._build_router_features
live = {}


def grab(self, **kw):  # the model's full router features (all text tokens + mask)
    f = orig(self, **kw)
    live.clear()
    live.update(f)
    return f


model._build_router_features = grab.__get__(model)
batch = default_collate([ds[i] for i in d["index"][:8].tolist()])
torch.manual_seed(int(d["index"][0]) * 1000 + 0)  # the seed the script used for this batch
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, logs = model.training_loss(batch)
cf = model._last_cf_cost.float().cpu()
dc = float((cf - d["costs"][:8]).abs().max())
dm = max(abs(float(cf[:, i].mean()) - logs[f"loss_action_{s}"]) for i, s in enumerate(ex.STREAMS))
assert dc < 1e-6 and dm < 1e-6, (dc, dm)
print(f"2. re-run with the same seed: costs identical (max diff {dc:.1e}); column means == logged "
      f"loss_action_* (max diff {dm:.1e})  OK", flush=True)

with torch.no_grad():
    r0 = pe(d["r0_latent"][:8].cuda().unsqueeze(2).to(pe.weight.dtype)).flatten(2).transpose(1, 2)
    cached = {"r0": r0, "text": d["text"][:8].cuda().unsqueeze(1), "text_mask": torch.ones(8, 1, device="cuda"),
              "proprio": d["proprio"][:8].cuda(), "act_hist": d["act_hist"][:8].cuda()}
    dl = float((router(live) - router(cached)).abs().max())
assert dl < 2e-2, dl
print(f"3. router(cached features) vs router(live features): max logit diff {dl:.1e}  OK", flush=True)

# 4. training + evaluate paths with the added line
model._build_router_features = orig.__get__(model)
model.modality_router.train()
model.zero_grad(set_to_none=True)
loss, _ = model.training_loss(default_collate([ds[i] for i in (0, 70000)]))
loss.backward()
n_router = sum(p.grad is not None for p in model.modality_router.parameters())
n_other = sum(p.grad is not None for p in model.parameters()) - n_router
assert n_router > 0 and n_other == 0 and model._last_cf_cost.shape == (2, 3)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    vl, _ = model.training_loss(Wan22Trainer._to_batched_eval_sample(ds[100000]))
assert torch.isfinite(vl) and model._last_cf_cost.shape == (1, 3)
print(f"4. router-only training step (router grads {n_router}, other {n_other}) and evaluate path (B=1) OK")

c = d["costs"]
print("   costs (first 4 samples, rgb/depth/flow):\n", c[:4].numpy())
print("   label = argmin per sample:", c.argmin(1).tolist())
print("ALL OK")
