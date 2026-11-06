"""Router with the past 32 executed actions (router_action_history: true, embed router), CPU, real weights + data.

  1. data: sample["action_history"] [K, A] = the normalized actions a_{t-K} .. a_{t-1} (== the first action of
     samples t-K .. t-1), the action window itself is unchanged, and at an episode start every row is pad (zeros);
  2. router module: act_proj only when enabled (no-history router keeps its old parameters), and the logits
     depend on the history;
  3. training_loss with the history (REINFORCE and counterfactual): the history is passed explicitly
     (training_loss -> _sample_modality_select -> _build_router_features), the router reads exactly the
     batch's history and gets a gradient (incl. act_proj), its features stay detached;
  3c. the trainer's eval path: training_loss on one sample under model.eval() (B=1), with and without history;
  4. eval buffer (ActionHistoryBuffer): the first replan_steps rows of each chunk, the last K, front-padded;
     infer_action(action_history=...) gives the router the same features as training.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_router_action_history.py
"""
import os
import types

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from experiments.libero.eval_infer import ActionHistoryBuffer
from fastwam.models.wan22.modality_router_new import EmbedModalityRouter
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_tok_embed_ahist"
RUN = "runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37"   # no router weights


def compose_task(task, *extra):
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        return compose(config_name="train", overrides=[f"task={task}", *extra])


def main():
    os.chdir(REPO)
    cfg = compose_task(TASK)
    K = int(cfg.data.train.action_history_steps)
    assert K == 32

    # 4a. buffer indexing (cheap, first)
    buf = ActionHistoryBuffer(types.SimpleNamespace(router_action_history=True, router_action_history_steps=K,
                                                    action_expert=types.SimpleNamespace(action_dim=2)), cfg)
    R = buf.replan
    assert buf.kwargs()["action_history_is_pad"].tolist() == [1.0] * K
    for c in range(12):                                  # chunk c: rows 100c + i, only the first R executed
        buf.push(torch.stack([torch.full((2,), 100.0 * c + i) for i in range(32)]).unsqueeze(0))
    kw = buf.kwargs()
    executed = [100.0 * c + i for c in range(12) for i in range(R)]
    assert kw["action_history"][:, 0].tolist() == executed[-K:] and kw["action_history_is_pad"].sum() == 0
    buf.rows = buf.rows[: 2 * R]
    kw = buf.kwargs()
    assert kw["action_history_is_pad"].tolist() == [1.0] * (K - 2 * R) + [0.0] * (2 * R)
    assert kw["action_history"][: K - 2 * R].abs().sum() == 0
    off = ActionHistoryBuffer(types.SimpleNamespace(), cfg)
    off.push(torch.zeros(1, 32, 7))
    assert off.kwargs() == {}
    print(f"4a. eval buffer: first {R} rows per chunk, last {K}, front-padded at the episode start; off = no kwargs  OK")

    # 1. data
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    ds0 = instantiate(compose_task(TASK, "data.train.action_history_steps=0").data.train,
                      pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    starts = ds.lerobot_dataset.episode_data_index["from"].tolist()
    i = starts[3] + 100                                  # 100 steps into episode 3
    s = ds._get(i)
    assert s["action_history"].shape == (K, 7) and s["action_history_is_pad"].shape == (K,)
    assert not s["action_history_is_pad"].any()
    assert torch.equal(s["action"], ds0._get(i)["action"]), "action window must not change"
    for k in (1, 2, 17, K):
        prev = ds0._get(i - k)["action"][0]
        assert torch.allclose(s["action_history"][K - k], prev, atol=1e-6), k
    s0 = ds._get(starts[3])
    assert s0["action_history_is_pad"].all() and s0["action_history"].abs().sum() == 0
    s5 = ds._get(starts[3] + 5)
    assert s5["action_history_is_pad"].tolist() == [True] * (K - 5) + [False] * 5
    print(f"1. data: action_history [K={K}, 7] == first action of samples t-{K} .. t-1; window unchanged; "
          "episode start all pad (zeros), t=5 -> 5 real rows  OK")

    # 2. router module
    r_old = EmbedModalityRouter(dim=16, num_classes=3)
    r_new = EmbedModalityRouter(dim=16, num_classes=3, action_history_dim=K * 8)
    assert not any("act_proj" in k for k in r_old.state_dict()) and r_old.head[2].in_features == 3 * 128
    assert r_new.head[2].in_features == 4 * 128
    r_new.eval()
    f = {"r0": torch.randn(2, 5, 16), "text": torch.randn(2, 4, 16), "text_mask": torch.ones(2, 4),
         "proprio": torch.randn(2, 16), "act_hist": torch.randn(2, K * 8)}
    l1 = r_new(f); f["act_hist"] = torch.randn(2, K * 8)
    assert (r_new(f) - l1).abs().max() > 1e-4
    print("2. router: act_proj only when enabled (old router unchanged), logits depend on the history  OK")

    # 3. training (REINFORCE, then counterfactual)
    batch = default_collate([ds._get(i), ds._get(starts[3] + 5)])
    for task in (TASK, TASK.replace("_ahist", "_cf_ahist")):
        mcfg = compose_task(task)
        model = instantiate(mcfg.model, model_dtype=torch.float32, device="cpu")
        model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
        model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
        Wan22Trainer._apply_dit_only_train_mode(model)
        model.router_warmup_steps = 0
        assert model.router_action_history and model.router_action_history_steps == K
        assert model.modality_router.action_history_dim == K * (7 + 1)
        cap = {}
        build = model._build_router_features

        def wrap(**kw):
            cap["in"], cap["feats"] = kw, build(**kw)
            return cap["feats"]
        model._build_router_features = wrap
        model.dit.train()
        torch.manual_seed(0)
        loss, logs = model.training_loss(batch)
        model._build_router_features = build
        ah = cap["feats"]["act_hist"]
        exp = torch.cat([batch["action_history"].flatten(1), batch["action_history_is_pad"].float()], 1)
        assert ah.shape == (2, K * 8) and torch.allclose(ah, exp), "router must read the batch history"
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = model.modality_router.act_proj[1].weight.grad
        assert g is not None and float(g.abs().sum()) > 0
        model.zero_grad(set_to_none=True)
        model.modality_router(build(**cap["in"])).logsumexp(-1).sum().backward()
        ve = model.video_expert
        assert ve.patch_embedding.weight.grad is None and ve.text_embedding[0].weight.grad is None
        print(f"3. {model.router_reward}: training_loss {float(loss):.4f}, router reads the batch history "
              "(rows + pad flags), act_proj gets grad, features detached  OK")

        # 3c. trainer eval path (B=1): _to_batched_eval_sample drops the history -> the router sees all pad
        model.eval()
        one = Wan22Trainer._to_batched_eval_sample(ds[i])
        assert "action_history" not in one
        with torch.no_grad():
            vl, _ = model.training_loss(one)
        assert torch.isfinite(vl)
        print(f"3c. Trainer.evaluate path (B=1, no history -> all pad): val loss {float(vl):.4f}  OK")

    # 4b. inference: same router features as training for the same history (last model = cf)
    train_ah = model._router_action_history_input(batch["action_history"][:1], batch["action_history_is_pad"][:1], 1, "cpu")
    # unbatched [K, A] (what infer_action gets from the eval buffer) == batched; none == all pad
    assert torch.equal(model._router_action_history_input(batch["action_history"][0], batch["action_history_is_pad"][0], 1, "cpu"), train_ah)
    none = model._router_action_history_input(None, None, 1, "cpu")
    assert none[:, : K * 7].abs().sum() == 0 and none[:, K * 7:].eq(1).all()
    cap = {}
    build = model._build_router_features

    def wrap2(**kw):
        cap["feats"] = build(**kw)
        return cap["feats"]
    model._build_router_features = wrap2
    with torch.no_grad():
        out = model.infer_action(prompt=None, input_image=batch["video_rgb"][0, :, 0], action_horizon=32,
                                 num_video_frames=int(batch["video_rgb"].shape[2]), proprio=batch["proprio"][0, 0],
                                 context=batch["context"][0], context_mask=batch["context_mask"][0],
                                 num_inference_steps=2, seed=0, action_history=batch["action_history"][0],
                                 action_history_is_pad=batch["action_history_is_pad"][0])
    model._build_router_features = build
    assert torch.allclose(cap["feats"]["act_hist"], train_ah) and out["active_future"] in model._enabled_streams()
    print(f"4b. infer_action(action_history=...): router features == training features, picked {out['active_future']}  OK")
    print("ALL ROUTER ACTION HISTORY CHECKS PASSED")


if __name__ == "__main__":
    main()
