import torch, torch.nn as nn, torch.nn.functional as F
from fastwam.models.wan22.modality_router_new import ModalityRouter
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew as M
from fastwam.models.wan22.fastwam_joint_multimodal_select_gumbel import FastWAMJointMultimodalSelectGumbel as G
torch.manual_seed(0)
B, C, H, W, L, D = 64, 4, 2, 2, 5, 8
def make(cls):
    m = cls.__new__(cls); nn.Module.__init__(m)
    m.modality_router = ModalityRouter(r0_dim=C*H*W, context_dim=D, context_len=L, num_classes=3)
    m.modality_select = "router"; m.stream_router_temperature = 1.0
    m.router_warmup_steps, m.router_grad_accum, m._router_calls = 3, 2, 0
    return m
r0 = torch.randn(B, C, H, W); ctx = torch.randn(B, L, D); mask = torch.ones(B, L, dtype=torch.bool)
# Bias the router hard toward stream 2 so "uniform" vs "policy" is easy to tell apart.
m = make(M)
with torch.no_grad(): m.modality_router.head[5].bias.copy_(torch.tensor([-5.0, -5.0, 5.0]))
m.modality_router.train()
log = []
for call in range(10):
    w = m._router_in_warmup()
    ch, lg, lp = m._sample_modality_select(("rgb", "depth", "flow"), B, torch.device("cpu"), r0_latent=r0, context=ctx, context_mask=mask)
    log.append((call, call // 2, w, round(float((ch == 2).float().mean()), 2)))
    m._router_tick()
print("1. REINFORCE (warmup=3 opt steps, grad_accum=2): (call, opt_step, warmup, frac stream2)")
for row in log: print("   ", row)
# 2. zero router grads in warm-up (x0 loss), nonzero after
m2 = make(M); m2.modality_router.train()
for flag in (True, False):
    m2.modality_router.zero_grad()
    lg = m2.modality_router({"r0": r0.flatten(1), "context": ctx, "context_mask": mask})
    loss = (0.0 if flag else 1.0) * lg.log_softmax(-1)[:, 0].mean()
    loss.backward(); g = m2.modality_router.head[5].weight.grad
    print(f"2. loss x{0.0 if flag else 1.0}: router grad exists {g is not None}, |grad| {float(g.abs().sum()):.4f}")
# 3. eval mode never in warm-up; argmax
m3 = make(M); m3.modality_router.eval()
print("3. eval: in_warmup", m3._router_in_warmup(), "| tick counts in eval?", (m3._router_tick(), m3._router_calls)[1])
# 4. resume clock from checkpoint step
m4 = make(M); m4._router_calls = int(1500) * m4.router_grad_accum; m4.router_warmup_steps = 2000; m4.modality_router.train()
print("4. resumed at step 1500 (warmup 2000): in_warmup", m4._router_in_warmup())
m4._router_calls = 2000 * 2; print("   resumed at step 2000: in_warmup", m4._router_in_warmup())
# 5. Gumbel warm-up gate: constant, uniform, tau clock not advanced
g = make(G); g.router_gumbel_tau_start, g.router_gumbel_tau_end, g.router_gumbel_tau_anneal_steps = 1.0, 0.5, 100
g.router_gumbel_hard = True; g._gumbel_step = 0; g._action_stream_log_gate = None
g.modality_router.train()
logits = g.modality_router({"r0": r0.flatten(1), "context": ctx, "context_mask": mask})
with torch.no_grad(): logits = logits + torch.tensor([-5.0, -5.0, 5.0])
logits.requires_grad_(True)
gate, ch, tau = g._router_log_gate(logits)
print(f"5. gumbel warm-up: gate requires_grad {gate.requires_grad}, frac stream2 {float((ch==2).float().mean()):.2f}, tau {tau}, gumbel_step {g._gumbel_step}")
g._router_calls = 3 * 2
gate, ch, tau = g._router_log_gate(logits)
print(f"   after warm-up: gate requires_grad {gate.requires_grad}, frac stream2 {float((ch==2).float().mean()):.2f}")
