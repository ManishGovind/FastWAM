"""Is each future slot supervised on its own modality? (label-free check of training targets)

Runs the real ``training_loss`` path at fixed noise levels sigma and, for every slot, turns the
predicted velocity into an x0 estimate (x0 = x_t - sigma * v) and compares it with the clean
future latents of ALL three streams. If the rgb slot were (wrongly) trained on depth, its x0
would be closest to the depth latent even at low noise.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/slot_supervision_check.py \
        [--task T --run R --ckpt step_020000.pt]   (default: random_full_all, step_021700.pt)
"""

import argparse
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="libero_joint_2cam224_1e-4_random_full_all")
    ap.add_argument("--run", default="runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37")
    ap.add_argument("--ckpt", default="step_021700.pt", help="file in <run>/checkpoints/weights/")
    ap.add_argument("--sigmas", default="0.2,0.5,0.8,0.95,1.0")
    ap.add_argument("--samples", default="0,90000,180000")
    args = ap.parse_args()
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(f"{args.run}/checkpoints/weights/{args.ckpt}")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.dit.eval()          # no dropout; training_loss path otherwise unchanged
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{args.run}/dataset_stats.json")
    batch = default_collate([ds._get(int(i)) for i in args.samples.split(",")])
    streams = model._enabled_streams()

    cap = {}
    enc, jd = model._encode_stream_latents, model._joint_denoise_multistream

    def enc_wrap(video, **kw):
        out = enc(video, **kw)
        cap.setdefault("clean", []).append(out[0])
        return out

    def jd_wrap(**kw):
        out = jd(**kw)
        cap["noisy"], cap["pred"] = kw["noisy_latents"], out[0]
        return out

    model._encode_stream_latents, model._joint_denoise_multistream = enc_wrap, jd_wrap
    print(f"task={args.task}  samples={args.samples}  streams={streams}")
    print("rel. MSE of each slot's x0 estimate (futures) vs each clean stream; * = closest")
    for sigma in (float(s) for s in args.sigmas.split(",")):
        cap.clear()
        model.train_video_scheduler.sample_training_t = (
            lambda batch_size, device, dtype, s=sigma: torch.full((batch_size,), s * 1000.0, device=device, dtype=dtype))
        torch.manual_seed(0)
        with torch.no_grad():
            model.training_loss(batch)
        clean = dict(zip(streams, cap["clean"]))
        row = []
        for slot in streams:
            x0 = (cap["noisy"][slot] - sigma * cap["pred"][slot])[:, :, 1:]
            errs = {m: float(((x0 - clean[m][:, :, 1:]) ** 2).mean() / (clean[m][:, :, 1:] ** 2).mean()) for m in streams}
            best = min(errs, key=errs.get)
            row.append(f"{slot:5s} slot: " + " ".join(f"{m}={e:.3f}{'*' if m == best else ' '}" for m, e in errs.items()))
        print(f"sigma={sigma:.2f} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
