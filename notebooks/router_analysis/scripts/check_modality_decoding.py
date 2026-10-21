"""Does the modality condition make the model generate the requested modality?

For each validation window, the same input (R0 image, instruction, proprio, noise seed)
is generated three times with ``active_future`` = rgb / depth / flow. With
``modality_prompt`` on, the video expert reads "... Predict the <stream> video.". Each
decoded future is compared with the ground-truth RGB / depth / flow futures of that
window (future frames only; frame 0 is the shared RGB image).

Meaningful condition  =  the diagonal of the 3x3 table (generated X vs GT X) beats the
off-diagonal in every row. Without a condition (old router checkpoint) the three
generations differ only by noise seed, so no clear diagonal is expected.

Run from the repo root (CPU keeps it off the GPUs used by training):
    CUDA_VISIBLE_DEVICES="" python notebooks/router_analysis/scripts/check_modality_decoding.py \\
        --task libero_joint_2cam224_1e-4_router_short_m_context \\
        --run runs/libero_joint_2cam224_1e-4_router_short_m_context/2026-10-08_20-10-53 \\
        --step 10000 --n 9 --steps 10 --device cpu
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

REPO = "/data/mgovind/FastWAM"
STREAMS = ("rgb", "depth", "flow")


def gt01(video: torch.Tensor) -> torch.Tensor:
    """[3, T, H, W] in (-1, 1) -> [0, 1] (the trainer's evaluate() convention)."""
    return ((video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", required=True, help="run dir with checkpoints/weights and dataset_stats.json")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n", type=int, default=9, help="validation windows (spread over the dataset)")
    ap.add_argument("--steps", type=int, default=10, help="denoising steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--videos", type=int, default=3, help="save grid videos for the first N windows")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, help="bf16 / fp32 (default: bf16 on cuda, fp32 on cpu)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-task-mismatch", action="store_true", help="use --task even if the run was trained with another")
    a = ap.parse_args()
    os.chdir(REPO)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype or ("bf16" if a.device == "cuda" else "fp32")]
    out = a.out or f"notebooks/router_analysis/outputs/modality_decoding/{os.path.basename(a.run.rstrip('/'))}_step{a.step:06d}"
    os.makedirs(out, exist_ok=True)

    # The task must match the run: a wrong task builds the model with other options (e.g.
    # modality_prompt off), and the "condition" silently does nothing.
    run_task = OmegaConf.load(f"{a.run}/config.yaml").wandb.name
    if a.task != run_task and not a.allow_task_mismatch:
        raise SystemExit(f"--task {a.task} != the run's task {run_task} (pass --allow-task-mismatch to force)")
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={a.task}"])
    model = instantiate(cfg.model, model_dtype=dt, device=a.device)
    model.load_checkpoint(f"{a.run}/checkpoints/weights/step_{a.step:06d}.pt")
    model = model.to(a.device).eval()
    print(f"model {type(model).__name__} | modality_prompt={getattr(model, 'modality_prompt', False)} "
          f"mode={getattr(model, 'modality_prompt_mode', '-')} target={getattr(model, 'modality_prompt_target', '-')} "
          f"| device={a.device} dtype={dt}", flush=True)

    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{a.run}/dataset_stats.json")
    idxs = np.linspace(0, len(ds) - 1, a.n).round().astype(int)
    psnr = np.full((a.n, 3, 3), np.nan)
    ssim = np.full((a.n, 3, 3), np.nan)
    t0 = time.time()
    for w, i in enumerate(idxs):
        s = ds[int(i)]
        gts = {name: gt01(s[f"video_{name}"]) for name in STREAMS}
        gens = {}
        for gi, gname in enumerate(STREAMS):
            with torch.no_grad():
                pred = model.infer_joint(
                    prompt=None,
                    input_image=s["video_rgb"][:, 0].to(a.device, dt),
                    num_video_frames=int(s["video_rgb"].shape[1]),
                    action_horizon=int(s["action"].shape[0]),
                    action=None,
                    proprio=s["proprio"][0].to(a.device),
                    context=s["context"].to(a.device),
                    context_mask=s["context_mask"].to(a.device),
                    num_inference_steps=a.steps,
                    seed=a.seed,
                    active_future=gname,
                    prompt_text=s["prompt"],
                )
            gen = pil_frames_to_video_tensor(pred["video"])  # [3, T, H, W] in [0, 1]
            gens[gname] = gen
            for ti, tname in enumerate(STREAMS):
                g, t = gen[:, 1:], gts[tname][:, 1:]  # future frames only
                psnr[w, gi, ti] = video_psnr(pred=g, target=t)
                ssim[w, gi, ti] = video_ssim(pred=g, target=t)
        print(f"window {w + 1}/{a.n} (idx {int(i)}) | PSNR rows=generated, cols=GT rgb/depth/flow:\n"
              + "\n".join(f"   gen {gn:5s} " + " ".join(f"{psnr[w, gi, ti]:6.2f}" for ti in range(3)) for gi, gn in enumerate(STREAMS))
              + f"   | {time.time() - t0:.0f}s", flush=True)
        if w < a.videos:  # grid: rows = generated rgb/depth/flow, then GT rgb/depth/flow; columns = frames
            rows = [gens[n] for n in STREAMS] + [gts[n] for n in STREAMS]
            grid = torch.cat([torch.cat([r[:, k] for r in rows], dim=1) for k in range(rows[0].shape[1])], dim=2)  # [3, 6H, T*W]
            frames = []
            for k in range(rows[0].shape[1]):
                stacked = torch.cat([r[:, k] for r in rows], dim=1)  # [3, 6H, W]
                frames.append(Image.fromarray((stacked.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)))
            save_mp4(frames, f"{out}/window{w:02d}_idx{int(i)}.mp4", fps=2)
            Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)).save(f"{out}/window{w:02d}_idx{int(i)}.png")

    mp, ms = np.nanmean(psnr, 0), np.nanmean(ssim, 0)
    diag_wins = [(psnr[:, gi, gi] >= np.max(np.delete(psnr[:, gi, :], gi, axis=1), axis=1)).mean() for gi in range(3)]
    print("\n=== mean PSNR (rows = generated with condition, cols = ground truth) ===")
    print(f"{'':12s}" + "".join(f"{f'GT {t}':>10s}" for t in STREAMS))
    for gi, gn in enumerate(STREAMS):
        print(f"gen {gn:8s}" + "".join(f"{mp[gi, ti]:10.2f}" for ti in range(3)) + f"   diagonal best in {100 * diag_wins[gi]:.0f}% of windows")
    print("\n=== mean SSIM ===")
    for gi, gn in enumerate(STREAMS):
        print(f"gen {gn:8s}" + "".join(f"{ms[gi, ti]:10.3f}" for ti in range(3)))
    json.dump({"task": a.task, "run": a.run, "step": a.step, "n": a.n, "steps": a.steps, "windows": idxs.tolist(),
               "psnr": psnr.tolist(), "ssim": ssim.tolist(), "mean_psnr": mp.tolist(), "mean_ssim": ms.tolist(),
               "diag_best_frac": diag_wins}, open(f"{out}/results.json", "w"), indent=1)
    print(f"\nsaved {out}/results.json + grids (rows: gen rgb/depth/flow, GT rgb/depth/flow)")


if __name__ == "__main__":
    main()
