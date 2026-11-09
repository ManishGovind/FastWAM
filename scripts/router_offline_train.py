"""Train the modality router offline on cached features + cf costs (scripts/router_offline_extract.py).

Losses (same router, same inputs; only the objective differs):
  expected : sum_m pi(m) C^m - beta*H(pi) + gamma*KL(U || mean pi)   (the online cf router loss)
  ce_hard  : -log pi(argmin_m cost^m)                                  (one-hot label per sample)
  ce_soft  : -sum_m q(m) log pi(m),  q = softmax(-C / T)
C = per-sample relative cost: cost - row mean, divided by the std of the micro-batch -- exactly as
online. Online steps are 16 micro-batches of 8 (4 GPUs x 4 accum), each with its own std and collapse
term, then averaged; we do the same with --group 8 inside a batch of 128.
Optimizer as online: AdamW lr 1e-4, wd 0.01, cosine with 5% warm-up, grad clip 1.0, ~10 epochs.

Output: <out>/router_<loss>.pt (router state + log) and, with --export, a full checkpoint
(cf+ahist experts + this router) for the standard LIBERO / LIBERO-Plus eval.

    PYTHONPATH=src:. python scripts/router_offline_train.py --loss expected --export
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fastwam.models.wan22.modality_router_new import EmbedModalityRouter, collapse_kl

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache/router_offline/cf_ahist_step021700"
EXPERTS = REPO / "checkpoints/router_ablation/cf_ahist_step021700_experts_only.pt"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=("expected", "ce_hard", "ce_soft"), required=True)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--experts", default=str(EXPERTS))
    p.add_argument("--out", default=str(REPO / "checkpoints/router_offline"))
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--group", type=int, default=8, help="online micro-batch size (per-group std + collapse)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=None, help="entropy coef (default 0.01 expected, 0 CE)")
    p.add_argument("--gamma", type=float, default=None, help="collapse-KL coef (default 0.1 expected, 0 CE)")
    p.add_argument("--soft-temp", type=float, default=1.0)
    p.add_argument("--val-every-block", type=int, default=20, help="hold out every N-th block of 2000 indices")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", action="store_true")
    return p.parse_args()


def load_cache(path):
    files = sorted(glob.glob(f"{path}/shard*_chunk*.pt"))
    assert files, f"no chunks in {path}"
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    data = {k: torch.cat([p[k] for p in parts]) for k in ("r0_latent", "text", "proprio", "act_hist", "costs", "index")}
    order = data["index"].argsort()
    return {k: v[order] for k, v in data.items()}, len(files)


def relative_cost(costs, group):
    """Online: C = cf - row mean, / std over the micro-batch (all its entries)."""
    c = costs - costs.mean(dim=1, keepdim=True)
    g = c.view(-1, group, c.shape[1])
    std = g.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    return (g / std).view_as(c)


def router_loss(logits, costs, args):
    """Mean over micro-batches of the per-micro-batch loss (online accumulation + DDP average)."""
    logp = F.log_softmax(logits, dim=-1)
    C = relative_cost(costs, args.group)
    if args.loss == "expected":
        obj = (logp.exp() * C).sum(-1)
    elif args.loss == "ce_hard":
        obj = -logp.gather(1, costs.argmin(1, keepdim=True)).squeeze(1)
    else:
        obj = -(torch.softmax(-C / args.soft_temp, dim=-1) * logp).sum(-1)
    ent = -(logp.exp() * logp).sum(-1)
    G = args.group
    coll = torch.stack([collapse_kl(lg) for lg in logits.view(-1, G, logits.shape[1])])
    per_group = obj.view(-1, G).mean(1) - args.beta * ent.view(-1, G).mean(1) + args.gamma * coll
    return per_group.mean(), ent.mean()


class Inputs:
    """Cached features -> router inputs (frozen video patch embedding applied on the fly, in bf16 as online)."""

    def __init__(self, experts, device):
        mot = torch.load(experts, map_location="cpu", weights_only=False, mmap=True)["mot"]
        w, b = mot["mixtures.video.patch_embedding.weight"], mot["mixtures.video.patch_embedding.bias"]
        self.pe = torch.nn.Conv3d(w.shape[1], w.shape[0], kernel_size=w.shape[2:], stride=w.shape[2:])
        self.pe.load_state_dict({"weight": w, "bias": b})
        self.pe = self.pe.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def __call__(self, data, idx):
        d = self.device
        lat = data["r0_latent"][idx].to(d, non_blocking=True)
        r0 = self.pe(lat.unsqueeze(2)).flatten(2).transpose(1, 2)
        n = len(idx)
        return {"r0": r0, "text": data["text"][idx].to(d).unsqueeze(1), "text_mask": torch.ones(n, 1, device=d),
                "proprio": data["proprio"][idx].to(d), "act_hist": data["act_hist"][idx].to(d)}


@torch.no_grad()
def evaluate(router, inputs, data, idx, args):
    router.eval()
    picks, regret, agree, losses = [], [], [], []
    for i in range(0, len(idx) - len(idx) % args.group, 1024):
        j = idx[i:i + 1024]
        j = j[: len(j) - len(j) % args.group]
        lg = router(inputs(data, j))
        c = data["costs"][j].to(lg.device)
        losses.append(float(router_loss(lg, c, args)[0]) * len(j))
        a = lg.argmax(1)
        picks.append(a.cpu())
        regret.append((c.gather(1, a[:, None]).squeeze(1) - c.min(1).values).cpu())
        agree.append((a == c.argmin(1)).float().cpu())
    router.train()
    p = torch.cat(picks)
    n = len(p)
    return {"loss": sum(losses) / n, "pick": [float((p == m).float().mean()) for m in range(3)],
            "agree_argmin": float(torch.cat(agree).mean()), "regret": float(torch.cat(regret).mean())}


def main():
    args = parse()
    if args.beta is None:
        args.beta = 0.01 if args.loss == "expected" else 0.0
    if args.gamma is None:
        args.gamma = 0.1 if args.loss == "expected" else 0.0
    torch.manual_seed(args.seed)
    dev = "cuda"
    t0 = time.time()
    data, nfiles = load_cache(args.cache)
    N = len(data["index"])
    block = data["index"] // 2000
    is_val = (block % args.val_every_block) == 0
    tr, va = torch.nonzero(~is_val).squeeze(1), torch.nonzero(is_val).squeeze(1)
    print(f"cache: {nfiles} files, {N} samples (train {len(tr)}, val {len(va)}) in {time.time() - t0:.0f}s", flush=True)
    lab = data["costs"].argmin(1)
    print("label (argmin cost) share rgb/depth/flow:", [round(float((lab == m).float().mean()), 3) for m in range(3)])

    inputs = Inputs(args.experts, dev)
    router = EmbedModalityRouter(dim=3072, num_classes=3, hidden=256, dropout=0.1, history=False,
                                 action_history_dim=data["act_hist"].shape[1]).to(dev).float()
    opt = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = int(args.epochs * len(tr) / args.batch)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    print(f"loss={args.loss} beta={args.beta} gamma={args.gamma} T={args.soft_temp} | {steps} steps, batch {args.batch}, "
          f"group {args.group}", flush=True)

    log, g = [], torch.Generator().manual_seed(args.seed)
    perm, pos = tr[torch.randperm(len(tr), generator=g)], 0
    for step in range(steps):
        if pos + args.batch > len(perm):
            perm, pos = tr[torch.randperm(len(tr), generator=g)], 0
        j = perm[pos:pos + args.batch]
        pos += args.batch
        logits = router(inputs(data, j))
        loss, ent = router_loss(logits, data["costs"][j].to(dev), args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps - 1:
            v = evaluate(router, inputs, data, va, args)
            rec = {"step": step, "train_loss": float(loss), "entropy": float(ent), **{f"val_{k}": x for k, x in v.items()}}
            log.append(rec)
            print(f"step {step:6d}  loss {float(loss):+.4f}  H {float(ent):.3f} | val loss {v['loss']:+.4f}  "
                  f"pick r/d/f {v['pick'][0]:.2f}/{v['pick'][1]:.2f}/{v['pick'][2]:.2f}  "
                  f"agree {v['agree_argmin']:.3f}  regret {v['regret']:.5f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"router_{args.loss}" + (f"_T{args.soft_temp:g}" if args.loss == "ce_soft" else "") + f"_seed{args.seed}"
    torch.save({"modality_router": router.state_dict(), "args": vars(args), "log": log}, out / f"{name}.pt")
    json.dump(log, open(out / f"{name}_log.json", "w"), indent=1)
    if args.export:
        ck = torch.load(args.experts, map_location="cpu", weights_only=False)
        ck["modality_router"] = {k: v.cpu() for k, v in router.state_dict().items()}
        ck["modality_select"] = "router"
        ck["step"] = steps
        torch.save(ck, out / f"{name}_full.pt")
        print("exported", out / f"{name}_full.pt")
    print(f"done in {time.time() - t0:.0f}s -> {out / name}.pt")


if __name__ == "__main__":
    main()
