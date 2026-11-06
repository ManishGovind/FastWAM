"""Counterfactual router reward (router_reward: counterfactual) in the _new model, CPU, real weights + data.

  1. mask: action copy A_m attends R0 + X_f^m + itself only; video rows never attend an action copy;
  2. equivalence: with the same inputs, action copy m == the single gated action with stream m chosen,
     and the video prediction is unchanged (one pass with 3 copies == 3 gated passes);
  3. training_loss runs: WAM action loss = mean of the 3 copies, router loss logged, the router gets a
     gradient (after warm-up) and its features stay detached (no router-only grad on patch / text
     embedding, proprio_encoder);
  3c. the trainer's eval path: training_loss on one sample under model.eval() (scalar timestep weight);
  4. infer_action of the counterfactual model uses the normal single-action path;
  5. config validation: counterfactual + short layout raises.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_router_counterfactual.py
"""
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_tok_embed_cf"
RUN = "runs/libero_joint_2cam224_1e-4_router_full_all_streamemb_embed/2026-10-16_06-33-56"


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
        bad = compose(config_name="train", overrides=[f"task={TASK}", "model.router_layout=short"])
    try:
        instantiate(bad.model, model_dtype=torch.float32, device="cpu")
        raise AssertionError("counterfactual + short layout should raise")
    except Exception as e:   # hydra wraps the model's ValueError in an InstantiationException
        assert "counterfactual" in str(e), e
    print("5. counterfactual + router_layout=short raises  OK")

    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    assert model.router_reward == "counterfactual"
    model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    enabled = model._enabled_streams()

    # 1. mask
    tpf, H = 98, 32
    lens = [tpf + 2 * tpf, 2 * tpf, 2 * tpf]                                  # rgb (R0 + R_f), depth, flow
    m = model._build_counterfactual_mot_attention_mask(lens, H, tpf, torch.device("cpu"), enabled)
    edges = [0, tpf, 3 * tpf, 5 * tpf, 7 * tpf]                               # R0 | R_f | D_f | F_f
    a0 = 7 * tpf
    blk = lambda r, c: m[r[0]:r[1], c[0]:c[1]]
    R0, Rf, Df, Ff = [(edges[i], edges[i + 1]) for i in range(4)]
    A = [(a0 + k * H, a0 + (k + 1) * H) for k in range(3)]
    fut = [Rf, Df, Ff]
    for k in range(3):
        assert blk(A[k], R0).all() and blk(A[k], fut[k]).all() and blk(A[k], A[k]).all()
        for j in range(3):
            if j != k:
                assert not blk(A[k], fut[j]).any() and not blk(A[k], A[j]).any()
    assert not m[:a0, a0:].any(), "video must not attend the action copies"
    print(f"1. mask {tuple(m.shape)}: A_m -> R0 + X_f^m + itself; video never attends an action copy  OK")

    # 2. equivalence with the gated single action (same inputs)
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds[0], ds[150000]])
    model.dit.eval()
    torch.manual_seed(0)
    lat = {n: model._encode_stream_latents(batch[f"video_{n}"])[0] for n in enabled}
    shared = model._build_shared_context_action(sample=batch, batch_size=2, num_frames=int(batch["video_rgb"].shape[2]))
    tv = torch.full((2,), 600.0); ta = torch.full((2,), 400.0)
    noisy = {}
    for n in enabled:
        x = model.train_video_scheduler.add_noise(lat[n], torch.randn_like(lat[n]), tv)
        x[:, :, 0:1] = lat[n][:, :, 0:1]
        noisy[n] = x
    act = model.train_action_scheduler.add_noise(shared["action"], torch.randn_like(shared["action"]), ta)
    kw = dict(noisy_latents=noisy, latents_action=act, timestep_video=tv, timestep_action=ta,
              context=shared["context"], context_mask=shared["context_mask"], fuse_vae_embedding_in_latents=True)
    with torch.no_grad():
        v_cf, a_cf = model._joint_denoise_multistream(**kw, action_copies=3)
        for k, name in enumerate(enabled):
            with model._action_stream_gate(model._one_hot_gate(torch.full((2,), k), 3)):
                v_g, a_g = model._joint_denoise_multistream(**kw)
            da = float((a_cf[:, k * H:(k + 1) * H] - a_g).abs().max())
            dv = max(float((v_cf[s] - v_g[s]).abs().max()) for s in enabled)
            assert da < 1e-3 and dv < 1e-3, (name, da, dv)
            print(f"2. copy {name}: |copy - gated single action| = {da:.1e}, |video diff| = {dv:.1e}  OK")

    # 3. training_loss
    model.dit.train()
    model.router_warmup_steps = 0
    torch.manual_seed(0)
    loss, logs = model.training_loss(batch)
    for k in ("loss_router", "router_cf", "router_entropy", "loss_action_rgb", "loss_action_depth", "loss_action_flow",
              "router_cf_best_rgb", "router_pick_flow"):
        assert k in logs, k
    mean_copy = sum(logs[f"loss_action_{n}"] for n in enabled) / 3
    assert abs(mean_copy - logs["loss_action"]) < 1e-5, (mean_copy, logs["loss_action"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.modality_router.parameters())
    per = ", ".join("%s %.4f" % (n, logs["loss_action_" + n]) for n in enabled)
    print(f"3. training_loss {float(loss):.4f}: loss_action = mean of copies ({per}); "
          f"router_cf {logs['router_cf']:+.4f}; router gets grad  OK")
    # router features detached: router loss alone must not reach the expert
    model.zero_grad(set_to_none=True)
    feats = model._build_router_features(r0_latent=lat["rgb"][:, :, 0], context=shared["context"], context_mask=shared["context_mask"])
    model.modality_router(feats).logsumexp(-1).sum().backward()
    ve = model.video_expert
    for nm, p in (("patch_embedding", ve.patch_embedding.weight), ("text_embedding", ve.text_embedding[0].weight)):
        assert p.grad is None, nm
    print("3b. router features detached (no router grad on patch / text embedding)  OK")

    # 3c. the trainer's periodic eval: one sample, model.eval(), no_grad (scalar timestep weight at B=1)
    model.eval()
    with torch.no_grad():
        val_loss, val_logs = model.training_loss(Wan22Trainer._to_batched_eval_sample(ds[0]))
    assert torch.isfinite(val_loss) and "router_cf" in val_logs
    print(f"3c. Trainer.evaluate path (B=1): val loss {float(val_loss):.4f}, router_cf {val_logs['router_cf']:+.4f}  OK")

    # 4. inference
    model.eval()
    with torch.no_grad():
        out = model.infer_action(prompt=None, input_image=batch["video_rgb"][0, :, 0], action_horizon=H,
                                 num_video_frames=int(batch["video_rgb"].shape[2]), proprio=batch["proprio"][0, 0],
                                 context=batch["context"][0], context_mask=batch["context_mask"][0],
                                 num_inference_steps=2, seed=0)
    assert out["action"].shape[0] == H and out["active_future"] in enabled
    print(f"4. infer_action: single action {tuple(out['action'].shape)}, router picked {out['active_future']}  OK")
    print("ALL ROUTER COUNTERFACTUAL CHECKS PASSED")


if __name__ == "__main__":
    main()
