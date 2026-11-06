"""Privileged MM teacher (FastWAMJointMultimodalPriv) + RGB distillation student (FastWAMJointDistill).

Weights: the MM+token step-21700 checkpoint stands in for the (not yet trained) priv teacher.
Checks:
  1. data: rgb_condition_modalities=[] -> video_depth/flow frame 0 are the real D0/F0, not RGB.
  2. priv mask matches the spec on a toy layout.
  3. priv == MM when every D0/F0 column is masked for the other tokens (exact up to bf16 noise):
     same action / future predictions -> the token layout, RoPE, stream embedding are right.
  4. priv uses D0/F0: changing D0/F0 changes the action prediction.
  5. priv training_loss: finite, backward reaches the stream embedding.
  6. student: teacher built lazily from a run dir, all teacher weights present in the checkpoint,
     not in the student's parameters / state_dict, no teacher grads; distill loss == manual value.
  7. Trainer.evaluate path (B=1, eval mode): training_loss + infer run.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/tests/test_distill_priv.py
"""

import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
MM_RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_rgb_depth_flow_streamemb_tok/2026-10-25_08-55-00"
MM_CKPT = f"{MM_RUN}/checkpoints/weights/step_021700.pt"
STATS = f"{MM_RUN}/dataset_stats.json"
FAKE_TEACHER = "/data/mgovind/claude_scratch/distill_test/teacher_run"
TEACHER_TASK = "libero_joint_2cam224_1e-4_rgb_depth_flow_streamemb_tok_priv"
STUDENT_TASK = "libero_joint_2cam224_1e-4_distill_mm_tok_priv"
DEV, DT = "cuda", torch.bfloat16


def cfg_for(task, *extra):
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        return compose(config_name="train", overrides=[f"task={task}", *extra])


def denoise_inputs(model, batch):
    """Noisy latents / action / timesteps for the priv-vs-MM comparisons (fixed seed)."""
    torch.manual_seed(0)
    noisy, fuse = {}, False
    t_v = torch.full((batch["action"].shape[0],), 500.0, device=DEV, dtype=DT)
    for n in model._enabled_streams():
        z, z0, fuse = model._encode_stream_latents(batch[f"video_{n}"])
        zt = model.train_video_scheduler.add_noise(z, torch.randn_like(z), t_v)
        zt[:, :, 0:1] = z0
        noisy[n] = zt
    shared = model._build_shared_context_action(batch, batch_size=batch["action"].shape[0], num_frames=int(batch["video"].shape[2]))
    a_t = torch.randn_like(shared["action"])
    t_a = torch.full_like(t_v, 700.0)
    return dict(noisy_latents=noisy, latents_action=a_t, timestep_video=t_v, timestep_action=t_a,
                context=shared["context"], context_mask=shared["context_mask"], fuse_vae_embedding_in_latents=fuse)


def main():
    os.chdir(REPO)
    tcfg = cfg_for(TEACHER_TASK)
    ds = instantiate(tcfg.data.train, pretrained_norm_stats=STATS)
    idx = [0, 90000, 180000, 270000]
    batch = default_collate([ds[i] for i in idx[:2]])

    # 1. real D0 / F0
    for n in ("depth", "flow"):
        d = (batch[f"video_{n}"][:, :, 0] - batch["video_rgb"][:, :, 0]).abs().max().item()
        assert d > 0.1, f"{n} frame 0 is still the RGB frame (max diff {d})"
    print("1. video_depth / video_flow frame 0 = real D0 / F0  OK", flush=True)

    teacher = instantiate(tcfg.model, model_dtype=DT, device=DEV)
    teacher.load_checkpoint(MM_CKPT)
    teacher = teacher.to(DEV)
    Wan22Trainer._apply_dit_only_train_mode(teacher)
    print(f"teacher class {type(teacher).__name__}", flush=True)

    # 2. mask spec on a toy layout: 3 streams x (1 cond + 2 future) tokens, tpf=1, 2 action tokens
    m = teacher._priv_mask([3, 3, 3], 2, 1, "cpu").int()
    R0, Rf, D0, Df, F0, Ff, A = [0], [1, 2], [3], [4, 5], [6], [7, 8], [9, 10]
    want = torch.zeros(11, 11, dtype=torch.int)
    for q, ks in [(R0, R0), (Rf, R0 + Rf), (D0, R0 + D0), (Df, R0 + D0 + Df), (F0, R0 + F0), (Ff, R0 + F0 + Ff), (A, list(range(11)))]:
        for i in q:
            want[i, ks] = 1
    assert torch.equal(m, want), f"\n{m}\nvs\n{want}"
    print("2. priv mask matches the per-stream spec  OK", flush=True)

    # 3. priv with D0/F0 hidden from everyone else == MM
    inp = denoise_inputs(teacher, batch)
    orig_mask = teacher._priv_mask

    def hide_cond(lens, action_len, tpf, device):
        mk = orig_mask(lens, action_len, tpf, device)
        for s in range(1, len(lens)):  # D0, F0 columns: only the token itself
            c = slice(sum(lens[:s]), sum(lens[:s]) + tpf)
            keep = mk[c, c].clone()
            mk[:, c] = False
            mk[c, c] = keep
        return mk

    with torch.no_grad():
        teacher._priv_mask = hide_cond
        pv_p, pa_p = teacher._joint_denoise_multistream(**inp)
        del teacher._priv_mask
        pv_m, pa_m = FastWAMJointMultimodal._joint_denoise_multistream(teacher, **inp)
    da = (pa_p.float() - pa_m.float()).abs().max().item() / pa_m.float().abs().max().item()
    dv = max(((pv_p[n][:, :, 1:].float() - pv_m[n][:, :, 1:].float()).abs().max() / pv_m[n].float().abs().max()).item() for n in pv_m)
    print(f"3. priv(D0/F0 hidden) vs MM: rel max diff action {da:.2e}, futures {dv:.2e}", flush=True)
    assert da < 2e-2 and dv < 2e-2

    # 4. D0 / F0 matter
    with torch.no_grad():
        _, pa = teacher._joint_denoise_multistream(**inp)
        inp2 = {**inp, "noisy_latents": {k: v.clone() for k, v in inp["noisy_latents"].items()}}
        for n in ("depth", "flow"):
            inp2["noisy_latents"][n][:, :, 0:1] = inp["noisy_latents"][n][:, :, 0:1].flip(0)  # swap D0/F0 across samples
        _, pa2 = teacher._joint_denoise_multistream(**inp2)
    d4 = (pa.float() - pa2.float()).abs().mean().item()
    print(f"4. swapping D0/F0 changes the action: mean |dv| = {d4:.4f}", flush=True)
    assert d4 > 1e-3

    # 5. priv training loss + backward
    teacher.zero_grad(set_to_none=True)
    loss, ld = teacher.training_loss(batch)
    loss.backward()
    g = teacher.video_expert.stream_embedding.grad
    assert torch.isfinite(loss) and g is not None and g.abs().sum() > 0
    print(f"5. priv training_loss {loss.item():.4f} {ld}, stream-emb grad {g.float().norm():.3e}  OK", flush=True)

    # fake teacher run dir: priv config + MM+tok weights
    os.makedirs(f"{FAKE_TEACHER}/checkpoints/weights", exist_ok=True)
    OmegaConf.save(tcfg, f"{FAKE_TEACHER}/config.yaml")
    link = f"{FAKE_TEACHER}/checkpoints/weights/step_021700.pt"
    if not os.path.exists(link):
        os.symlink(MM_CKPT, link)
    del teacher, g, loss
    torch.cuda.empty_cache()

    # 6. student
    scfg = cfg_for(STUDENT_TASK, f"model.teacher_run={FAKE_TEACHER}")
    student = instantiate(scfg.model, model_dtype=DT, device=DEV).to(DEV)
    Wan22Trainer._apply_dit_only_train_mode(student)
    sds = instantiate(scfg.data.train, pretrained_norm_stats=STATS)
    sbatch = default_collate([sds[i] for i in idx[:2]])
    assert student._teacher is None
    student.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    loss, ld = student.training_loss(sbatch)
    loss.backward()
    T = student._teacher
    assert T is not None and type(T).__name__ == "FastWAMJointMultimodalPriv"
    payload = torch.load(link, map_location="cpu")
    missing = set(T.mot.state_dict()) - set(payload["mot"])
    assert not missing, f"teacher weights missing from checkpoint: {sorted(missing)[:5]}"
    ok = all(torch.equal(T.mot.state_dict()[k].cpu(), payload["mot"][k].to(T.torch_dtype)) for k in list(payload["mot"])[:20])
    assert ok, "teacher weights differ from the checkpoint"
    tids = {id(p) for p in T.parameters()}
    assert not tids & {id(p) for p in student.parameters()}
    assert not any(k.startswith("_teacher") for k in student.state_dict())
    assert all(p.grad is None for p in T.parameters())
    n_grad = sum(p.grad is not None for p in student.parameters())
    assert torch.isfinite(loss) and n_grad > 0
    print(f"6a. student loss {loss.item():.4f} {ld}; teacher lazily built, weights == ckpt, "
          f"not in student params/state_dict, no teacher grads; {n_grad} student tensors with grad  OK", flush=True)

    # 6b. distill term == manual value (recompute with the same seed, record student pred + teacher target)
    rec = {}
    o_core, o_teach = student._joint_denoise_core, student._teacher_action

    def core(**k):
        out = o_core(**k)
        rec["pred_a"], rec["t_a"] = out[1].detach().float(), k["timestep_action"]
        return out

    def teach(*a):
        rec["v_t"] = o_teach(*a)
        return rec["v_t"]

    student._joint_denoise_core, student._teacher_action = core, teach
    with torch.no_grad():
        torch.manual_seed(1)
        loss_b, ld_b = student.training_loss(sbatch)
    del student._joint_denoise_core, student._teacher_action
    valid = (~sbatch["action_is_pad"].to(DEV)).float()
    gap = (((rec["pred_a"] - rec["v_t"]) ** 2).mean(-1) * valid).sum(1) / valid.sum(1).clamp(min=1)
    manual = (gap * student.train_action_scheduler.training_weight(rec["t_a"]).float()).mean().item()
    print(f"6b. loss_action {ld_b['loss_action']:.6f} vs manual {manual:.6f}", flush=True)
    assert abs(ld_b["loss_action"] - manual) < 1e-4 * max(1.0, abs(manual))

    # 7. Trainer.evaluate path: B=1, eval mode, training_loss + infer
    student.eval()
    sample = Wan22Trainer._to_batched_eval_sample(sds[idx[2]])
    with torch.no_grad(), torch.autocast("cuda", dtype=DT):
        vl, vd = student.training_loss(sample)
        pred = student.infer(
            prompt=None, input_image=sample["video"][0][:, 0].unsqueeze(0), num_frames=int(sample["video"].shape[2]),
            action=sample["action"][0], action_horizon=sample["action_horizon"], proprio=sample["proprio"][0, 0],
            context=sample["context"][0], context_mask=sample["context_mask"][0], text_cfg_scale=1.0,
            action_cfg_scale=1.0, num_inference_steps=4, seed=42, tiled=False,
        )
    assert torch.isfinite(vl) and pred["action"] is not None
    print(f"7. evaluate path (B=1): val loss {vl.item():.4f} {vd}; infer action {tuple(pred['action'].shape)}  OK", flush=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
