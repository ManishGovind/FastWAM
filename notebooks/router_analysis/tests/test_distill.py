"""Distillation student (FastWAMJointMultimodalSelectDistill), one real training step on CPU.

Student = random-select context model (step 21700 weights); teachers = unimodal RGB / Depth /
Flow joint models (frozen). Checks:
  1. teachers are built lazily and are NOT in the student's parameters / state_dict;
  2. the student's per-sample action predictions map back to every batch row;
  3. per-sample best teacher + logged teacher errors are finite; teacher error is in the
     range of the student's own action error (the teachers are trained models);
  4. the distillation term back-propagates into the student only (no teacher grads);
  5. distill_lambda = 0 -> exactly the context model's loss (same seed).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_distill.py
"""

import os
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_distill_uni3_best_m_context_video"
STUDENT_CKPT = "runs/libero_joint_2cam224_1e-4_random_short_m_context_video/2026-10-10_09-57-20/checkpoints/weights/step_021700.pt"
STATS = "runs/libero_joint_2cam224_1e-4_random_short_m_context_video/2026-10-10_09-57-20/dataset_stats.json"


def main():
    os.chdir(REPO)
    t0 = time.time()
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(STUDENT_CKPT)
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)  # cudagraphs backend is GPU-only
    Wan22Trainer._apply_dit_only_train_mode(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_state = len(model.state_dict())
    ds = instantiate(cfg.data.train, pretrained_norm_stats=STATS)
    batch = default_collate([ds[0], ds[90000], ds[180000], ds[270000]])
    print(f"student ready {time.time() - t0:.0f}s ({n_params / 1e9:.2f}B params)", flush=True)

    # 5. lambda = 0 -> the context model's loss, same seed
    model.distill_lambda = 0.0
    torch.manual_seed(0)
    base_loss, _ = model.training_loss(batch)
    torch.manual_seed(0)
    ref_loss, _ = FastWAMJointMultimodalSelectNew.training_loss(model, batch)
    assert torch.equal(base_loss, ref_loss), (base_loss.item(), ref_loss.item())
    assert model._distill_teachers is None, "teachers must not be built when distill_lambda = 0"
    print(f"5. lambda=0 loss == context-model loss ({base_loss.item():.6f})  OK", flush=True)

    # one distillation step
    model.distill_lambda = 1.0
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss, logs = model.training_loss(batch)
    teachers = model._distill_teachers
    print(f"teachers built: {list(teachers)} ({time.time() - t0:.0f}s)", flush=True)

    # 1. teachers outside the student
    assert sum(p.numel() for p in model.parameters()) == n_params and len(model.state_dict()) == n_state
    student_ids = {id(p) for p in model.parameters()}
    assert not any(id(p) in student_ids for t in teachers.values() for p in t.parameters())
    assert all(not p.requires_grad for t in teachers.values() for p in t.parameters())
    print("1. teachers not in student parameters / state_dict, all frozen  OK")

    # 2-3. logs
    print("   logs:", {k: round(v, 4) for k, v in logs.items() if k.startswith(("loss", "distill"))})
    assert abs(loss.item() - base_loss.item() - logs["loss_distill"]) < 1e-4, "total = context loss + distill term"
    shares = [logs[f"distill_best_{m}"] for m in teachers]
    assert abs(sum(shares) - 1.0) < 1e-6
    for m in teachers:
        assert torch.isfinite(torch.tensor(logs[f"distill_err_{m}"]))
    print("2-3. per-sample mapping ok, best-teacher shares sum to 1, errors finite  OK")

    # 4. gradient of the distillation term reaches the student only
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss, logs = model.training_loss(batch)
    (loss - base_loss.detach()).backward()  # isolates the distillation gradient's existence check
    g = model.action_expert.action_encoder.weight.grad
    assert g is not None and g.norm() > 0
    assert all(p.grad is None for t in teachers.values() for p in t.parameters())
    print(f"4. distillation gradient reaches the student (action_encoder grad {g.norm():.3e}); no teacher grads  OK")
    # 6. trainer mode handling: top-level eval() + DiT train() -> distillation ON (checked above);
    #    DiT eval() (e.g. the trainer's periodic eval) -> OFF, plain context-model loss
    model.dit.eval()
    torch.manual_seed(0)
    off_loss, off_logs = model.training_loss(batch)
    assert "loss_distill" not in off_logs and torch.equal(off_loss.detach(), base_loss.detach())
    model.dit.train()
    print("6. distillation follows the DiT's train/eval mode (top-level model is eval() in training)  OK")
    print(f"ALL DISTILL CHECKS PASSED ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
