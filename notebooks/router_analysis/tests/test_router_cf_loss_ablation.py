"""Router-loss ablation (router_cf_loss = expected | ce_hard | ce_soft) + train_router_only, GPU, real weights + data.

Init = checkpoints/router_ablation/cf_ahist_step021700_experts_only.pt (cf+ahist step 21700 minus the router).
  1. init: experts == the cf+ahist checkpoint, router freshly initialized (not the trained one), warm-up clock 0.
  2. train_router_only (trainer's _apply_dit_only_train_mode): only router params require grad; DiT stays in
     train mode; backward reaches the router only.
  3. per loss, same seed: everything except the router loss is identical across the three losses;
     cost term obj = loss_router + beta*H - gamma*KL: expected == router_cf; with a uniform router (output
     layer zeroed) expected == 0 and both CE losses == log 3 exactly.
  4. Trainer.evaluate path (eval mode, B=1, scalar timestep weight) for every loss.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/tests/test_router_cf_loss_ablation.py
"""
import math
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_tok_embed_cf_ahist_ronly_{}"
RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_router_full_all_streamemb_tok_embed_cf_ahist/2026-11-01_00-27-41"
DEV, DT = "cuda", torch.bfloat16
os.chdir(REPO)

with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfgs = {m: compose(config_name="train", overrides=[f"task={TASK.format(m)}"]) for m in ("expected", "ce_hard", "ce_soft")}
for m, c in cfgs.items():
    assert c.model.router_cf_loss == m and c.train_router_only and c.model.router_warmup_steps == 0
cfg = cfgs["expected"]
model = instantiate(cfg.model, model_dtype=DT, device=DEV)
model.load_checkpoint(cfg.resume)
model = model.to(DEV)

# 1. init
full = torch.load(f"{RUN}/checkpoints/weights/step_021700.pt", map_location="cpu", weights_only=False)
sd = model.mot.state_dict()
assert all(torch.equal(sd[k].cpu(), full["mot"][k]) for k in list(full["mot"])[::50])
r_trained = full["modality_router"]
r_now = model.modality_router.state_dict()
assert set(r_now) == set(r_trained) and not any(torch.equal(r_now[k].cpu(), r_trained[k]) for k in r_now if r_now[k].numel() > 1)
assert int(getattr(model, "_router_calls", 0)) == 0
print("1. experts == cf+ahist ckpt, router fresh (all tensors differ from the trained router), warm-up clock 0  OK", flush=True)
del full

# 2. router-only training mode
model.train_router_only = True
Wan22Trainer._apply_dit_only_train_mode(model)
assert model.dit.training and not any(p.requires_grad for p in model.dit.parameters())
assert all(p.requires_grad for p in model.modality_router.parameters())
assert not any(p.requires_grad for p in model.proprio_encoder.parameters())
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
batch = default_collate([ds[i] for i in (0, 70000, 140000, 210000)])

beta, gamma = model.router_entropy_coef, model.router_collapse_kl
res = {}
for mode in ("expected", "ce_hard", "ce_soft"):
    model.router_cf_loss = mode
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    loss, logs = model.training_loss(batch)
    loss.backward()
    mem = torch.cuda.max_memory_allocated() / 2**30
    n_router = sum(p.grad is not None for p in model.modality_router.parameters())
    n_other = sum(p.grad is not None for p in model.parameters()) - n_router
    obj = logs["loss_router"] + beta * logs["router_entropy"] - gamma * logs["router_collapse_kl"]
    res[mode] = (float(loss) - logs["loss_router"], obj, logs)
    print(f"   {mode:8s}: loss {float(loss):.4f}  cost-term {obj:+.4f}  H {logs['router_entropy']:.4f}  "
          f"router grads {n_router}, other grads {n_other}, peak {mem:.1f} GiB", flush=True)
    assert n_router > 0 and n_other == 0
print("2. only the router gets gradients (DiT frozen, still in train mode)  OK", flush=True)

# 3. only the router loss differs; exact values with a uniform router (output layer zeroed -> pi = 1/3):
#    expected = sum_m C^m / 3 = 0 (C is relative, mean 0 per sample); ce_hard = ce_soft = log 3.
base = res["expected"][0]
assert all(abs(r[0] - base) < 1e-4 * max(1, abs(base)) for r in res.values()), {k: v[0] for k, v in res.items()}
assert abs(res["expected"][1] - res["expected"][2]["router_cf"]) < 1e-4
saved = {k: v.clone() for k, v in model.modality_router.state_dict().items()}
with torch.no_grad():
    model.modality_router.head[-1].weight.zero_()
    model.modality_router.head[-1].bias.zero_()
uni = {}
for mode in ("expected", "ce_hard", "ce_soft"):
    model.router_cf_loss = mode
    torch.manual_seed(0)
    with torch.no_grad():
        _, lg = model.training_loss(batch)
    uni[mode] = lg["loss_router"] + beta * lg["router_entropy"] - gamma * lg["router_collapse_kl"]
    assert abs(lg["router_entropy"] - math.log(3)) < 1e-3
model.modality_router.load_state_dict(saved)
assert abs(uni["expected"]) < 1e-3, uni
assert all(abs(uni[m] - math.log(3)) < 1e-3 for m in ("ce_hard", "ce_soft")), uni
print(f"3. non-router loss identical across losses ({base:.4f}); uniform router: expected {uni['expected']:+.5f} (0), "
      f"ce_hard {uni['ce_hard']:.5f}, ce_soft {uni['ce_soft']:.5f} (log 3 = {math.log(3):.5f})  OK", flush=True)

# 4. Trainer.evaluate path
model.eval()
smp = Wan22Trainer._to_batched_eval_sample(ds[100000])
for mode in ("expected", "ce_hard", "ce_soft"):
    model.router_cf_loss = mode
    with torch.no_grad(), torch.autocast("cuda", dtype=DT):
        vl, vlogs = model.training_loss(smp)
    assert torch.isfinite(vl)
    print(f"   eval {mode:8s}: val loss {float(vl):.4f}", flush=True)
print("4. Trainer.evaluate path (B=1) for every loss  OK")
print("ALL OK")
