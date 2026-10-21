import torch, torch.nn as nn
from fastwam.models.wan22.modality_router_new import ModalityRouter, collapse_kl
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew as M

torch.manual_seed(0)
B, C, H, W, L, D = 4, 48, 14, 28, 129, 64
r = ModalityRouter(r0_dim=C*H*W, context_dim=D, context_len=L, num_classes=3).eval()
r0 = torch.randn(B, C*H*W); ctx = torch.randn(B, L, D)
mask = torch.ones(B, L, dtype=torch.bool); mask[:, 20:128] = False
a = r({"r0": r0, "context": ctx, "context_mask": mask})
ctx2 = ctx.clone(); ctx2[:, 20:128] = torch.randn(B, 108, D) * 100
b = r({"r0": r0, "context": ctx2, "context_mask": mask})
print("1. pad tokens ignored:", torch.allclose(a, b))

# Fake model: just the attributes the router paths use.
m = M.__new__(M); nn.Module.__init__(m)
m.modality_router = r.train(); m.modality_select = "router"; m.stream_router_temperature = 2.0
m.router_baseline = "timestep_ema"; m.router_baseline_bins = 4; m.router_baseline_momentum = 0.5
m._router_baseline_ema = None; m._router_baseline_seen = None

prop = nn.Linear(8, D)  # stands in for proprio_encoder
ctx_g = torch.cat([ctx[:, :-1], prop(torch.randn(B, 1, 8))], 1)
m.train()  # router.train() via recursion
ch, lg, lp = m._sample_modality_select(("rgb", "depth", "flow"), B, torch.device("cpu"),
    r0_latent=r0.view(B, C, H, W), context=ctx_g, context_mask=mask)
print("2. tempered: log_probs == log_softmax(logits):", torch.allclose(lp, lg.log_softmax(-1)))
(lp.gather(1, ch[:, None]).mean() + collapse_kl(lg)).backward()
print("3. no grad into proprio encoder:", prop.weight.grad is None,
      "| router has grad:", r.head[5].weight.grad is not None)

rew = torch.tensor([1., 2., 10., 20.]); sig = torch.tensor([0.1, 0.1, 0.9, 0.9])
b1 = m._router_reward_baseline(rew, sig)
print("4. first call falls back to batch mean:", b1.tolist())
b2 = m._router_reward_baseline(rew, sig)
print("5. second call uses per-bucket EMA:", b2.tolist(), "ema:", m._router_baseline_ema.tolist())
m.modality_router.eval(); before = m._router_baseline_ema.clone()
m._router_reward_baseline(rew * 5, sig)
print("6. eval does not update EMA:", torch.equal(before, m._router_baseline_ema))
m.router_baseline = "batch"
print("7. batch baseline:", m._router_reward_baseline(rew, sig).tolist())

# Trainer setup: top-level eval, router train (trainer._apply_dit_only_train_mode).
m.eval(); m.modality_router.train()
torch.manual_seed(1)
picks = torch.stack([m._sample_modality_select(("rgb","depth","flow"), B, torch.device("cpu"),
    r0_latent=r0.view(B, C, H, W), context=ctx, context_mask=mask)[0] for _ in range(50)])
print("8. trainer mode (self.training=%s, router.training=%s) samples, not argmax:"
      % (m.training, m.modality_router.training), bool((picks != picks[0]).any()))
m.modality_router.eval()
picks = torch.stack([m._sample_modality_select(("rgb","depth","flow"), B, torch.device("cpu"),
    r0_latent=r0.view(B, C, H, W), context=ctx, context_mask=mask)[0] for _ in range(5)])
print("9. router eval -> deterministic argmax:", bool((picks == picks[0]).all()))
