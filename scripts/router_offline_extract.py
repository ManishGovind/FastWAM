"""Offline router data: router input features + per-sample cf costs [B, 3] from a frozen cf model.

One normal training forward per batch (one noise draw), exactly as in online cf training:
  * costs = model._last_cf_cost  [B, 3] = lambda * action_mse * w(sigma_a) of each action copy
    (rgb, depth, flow) -- the same tensor the online router loss uses;
  * the router's inputs, recorded from model._build_router_features (wrapped at runtime):
    R0 VAE latent, text-embedding mean (= the router's own masked mean), proprio token,
    action history (+ pad flags).

Sharded and resumable: shard s of S takes samples s, s+S, ...; every --chunk samples are written to
  <out>/shard{s:02d}_chunk{c:05d}.pt  (existing chunks are skipped on restart).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/router_offline_extract.py --shard 0 --num-shards 4
"""
import argparse
import os
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset, default_collate

from fastwam.trainer import Wan22Trainer

REPO = Path(__file__).resolve().parents[1]
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_tok_embed_cf_ahist"
RUN = REPO / "runs" / TASK / "2026-11-01_00-27-41"
STREAMS = ("rgb", "depth", "flow")
# Local copy of the LIBERO data (identical to /net/labs/sdas24_lab/libero_mujoco3.3.2, which stalls).
LOCAL_DIRS = [str(REPO / f"data/libero_mujoco3.3.2/libero_{s}_no_noops_lerobot") for s in ("spatial", "object", "goal", "10")]
FEATS = ("r0_latent", "text", "proprio", "act_hist")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=TASK)
    p.add_argument("--ckpt", default=str(RUN / "checkpoints/weights/step_021700.pt"))
    p.add_argument("--stats", default=str(RUN / "dataset_stats.json"))
    p.add_argument("--out", default=str(REPO / "cache/router_offline/cf_ahist_step021700"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N samples of this shard")
    p.add_argument("--dataset-dirs", nargs="+", default=LOCAL_DIRS, help="override data.train.dataset_dirs")
    p.add_argument("--num-workers", type=int, default=6)
    return p.parse_args()


def load(args):
    t0 = time.time()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={args.task}",
                                                      "data.train.dataset_dirs=[" + ",".join(args.dataset_dirs) + "]"])
    assert cfg.model.router_reward == "counterfactual", "needs the cf model (one action copy per stream)"
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    print(f"[load] model built {time.time() - t0:.0f}s", flush=True)
    model.load_checkpoint(args.ckpt)
    model = model.to("cuda")
    print(f"[load] checkpoint loaded {time.time() - t0:.0f}s", flush=True)
    # Same modes as router-only training: DiT/proprio frozen in train mode; router in eval mode
    # (argmax, no warm-up clock) -- the costs do not depend on the router's choice (all copies run).
    model.train_router_only = True
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.modality_router.eval()
    ds = instantiate(cfg.data.train, pretrained_norm_stats=args.stats)
    print(f"[load] dataset ready ({len(ds)} samples) {time.time() - t0:.0f}s", flush=True)
    return model, ds


def record_features(model):
    """Wrap model._build_router_features (this instance only) to keep what the router reads."""
    orig = model._build_router_features
    box = {}

    def wrapped(**kw):
        feats = orig(**kw)
        m = feats["text_mask"].float().unsqueeze(-1)
        box["r0_latent"] = kw["r0_latent"].detach().to(torch.bfloat16).cpu()
        box["text"] = ((feats["text"].float() * m).sum(1) / m.sum(1).clamp_min(1.0)).to(torch.bfloat16).cpu()
        box["proprio"] = feats["proprio"].detach().to(torch.bfloat16).cpu()
        box["act_hist"] = feats["act_hist"].detach().float().cpu()
        return feats

    model._build_router_features = wrapped
    return box


@torch.no_grad()
def run_batch(model, batch, box, seed):
    model._last_cf_cost = None
    torch.manual_seed(seed)  # one noise draw per batch, reproducible
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.training_loss(batch)
    costs = model._last_cf_cost
    assert costs is not None and costs.shape[1] == len(STREAMS), "no cf cost recorded (is this the cf model?)"
    return {k: box[k] for k in FEATS}, costs.float().cpu()


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ds = load(args)
    box = record_features(model)
    idx = list(range(args.shard, len(ds), args.num_shards))
    if args.limit:
        idx = idx[: args.limit]
    chunks = [idx[i:i + args.chunk] for i in range(0, len(idx), args.chunk)]
    print(f"shard {args.shard}/{args.num_shards}: {len(idx)} samples, {len(chunks)} chunks, B={args.batch_size}", flush=True)
    for c, cidx in enumerate(chunks):
        path = out / f"shard{args.shard:02d}_chunk{c:05d}.pt"
        if path.exists():
            continue
        loader = DataLoader(Subset(ds, cidx), batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=default_collate)
        rec = {k: [] for k in (*FEATS, "costs")}
        t0 = time.time()
        for b, batch in enumerate(loader):
            feats, costs = run_batch(model, batch, box, seed=cidx[0] * 1000 + b)
            if b % 25 == 0:
                print(f"    batch {b}: {(b + 1) * args.batch_size / (time.time() - t0):.2f} samples/s", flush=True)
            for k in FEATS:
                rec[k].append(feats[k])
            rec["costs"].append(costs)
        data = {k: torch.cat(v) for k, v in rec.items()}
        data["index"] = torch.tensor(cidx)
        data["meta"] = {"task": args.task, "ckpt": args.ckpt, "streams": STREAMS,
                        "costs": "[N, 3] = loss_lambda_action * action_mse * w(sigma_a) per action copy, one noise draw"}
        tmp = path.with_suffix(".tmp")
        torch.save(data, tmp)
        os.replace(tmp, path)
        print(f"  chunk {c}: {len(cidx)} samples in {time.time() - t0:.0f}s "
              f"({len(cidx) / (time.time() - t0):.1f} samples/s) -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
