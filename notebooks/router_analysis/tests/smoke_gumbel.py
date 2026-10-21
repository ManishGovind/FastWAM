import torch, torch.nn as nn, torch.nn.functional as F
DEV = __import__("os").environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # TEST_DEVICE=cpu to avoid GPUs
from fastwam.models.wan22.fastwam_joint_multimodal_select_gumbel import FastWAMJointMultimodalSelectGumbel as G
torch.manual_seed(0)
dev = DEV
m = G.__new__(G); nn.Module.__init__(m)
m.enabled_video_streams = ("rgb", "depth", "flow"); m.torch_dtype = torch.bfloat16
m.modality_router = nn.Linear(4, 3).to(dev)           # stand-in: only .training matters for the gate
m.router_gumbel_tau_start, m.router_gumbel_tau_end, m.router_gumbel_tau_anneal_steps = 1.0, 0.5, 100
m.router_gumbel_hard = True; m._gumbel_step = 0; m._action_stream_log_gate = None
print("enabled:", m._enabled_streams())

tpf, F_rgb, F_fut, A = 3, 3, 2, 4                      # rgb: 3 frames (R0 + 2 future), futures: 2 frames
lens = [tpf * F_rgb, tpf * F_fut, tpf * F_fut]; S = sum(lens) + A
B = 3
logits = torch.randn(B, 3, device=dev, requires_grad=True)
m.modality_router.train()
log_gate, choices, tau = m._router_log_gate(logits)
print("1. train: tau", tau, "choices", choices.tolist(), "forward gate", log_gate.detach().tolist())
with m._action_stream_gate(log_gate):
    mask = m._build_multistream_mot_attention_mask(lens, A, tpf, torch.device(dev), "all")
print("   mask", tuple(mask.shape), mask.dtype, "| gate cleared:", m._action_stream_log_gate is None)

# Expected visible columns for action rows: R0 frame0, chosen future, action.
segs = {"r0": range(0, tpf), "rgb": range(tpf, lens[0]), "depth": range(lens[0], lens[0] + lens[1]),
        "flow": range(lens[0] + lens[1], sum(lens)), "act": range(sum(lens), S)}
ok = True
for b, c in enumerate(choices.tolist()):
    row = mask[b, sum(lens)].float()
    vis = set(torch.nonzero(row > -1e3).flatten().tolist())
    want = set(segs["r0"]) | set(segs[("rgb", "depth", "flow")[c]]) | set(segs["act"])
    ok &= vis == want
    vrows_same = torch.equal(mask[b, :sum(lens)], mask[0, :sum(lens)])
    ok &= vrows_same
print("2. action rows see exactly R0 + chosen future + action; video rows ungated:", ok)

# 3. Toy attention: full gated sequence vs short [R0 | chosen | A] sequence (inference path).
H, D = 4, 64
x = torch.randn(B, S, H * D, device=dev, dtype=torch.bfloat16)
Wq, Wk, Wv = (torch.randn(H * D, H * D, device=dev, dtype=torch.bfloat16) * 0.05 for _ in range(3))
def attn(xx, mm):
    q, k, v = (t.view(xx.shape[0], xx.shape[1], H, D).transpose(1, 2) for t in (xx @ Wq, xx @ Wk, xx @ Wv))
    if mm.ndim == 3: mm = mm[:, None]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mm).transpose(1, 2).reshape(xx.shape[0], xx.shape[1], -1)
full = attn(x, mask.detach())
maxdiff = 0.0
for b, c in enumerate(choices.tolist()):
    keep = list(segs["r0"]) + (list(segs["rgb"]) if c == 0 else list(segs[("rgb", "depth", "flow")[c]])) + list(segs["act"])
    xs = x[b:b + 1, keep]
    ms = torch.ones(len(keep), len(keep), dtype=torch.bool, device=dev)   # action rows: everything kept is visible
    short = attn(xs, ms)
    maxdiff = max(maxdiff, float((full[b, sum(lens):] - short[0, -A:]).abs().max()))
print("3. action output, gated full seq vs short seq: max |diff| =", maxdiff)

# 4. Gradient: see smoke_gumbel_real.py (real head dim / seq len).
# 5. Tau schedule + eval path.
taus = []
for s in (0, 50, 100, 200):
    m._gumbel_step = s; taus.append(round(m._gumbel_tau(), 4))
print("5. tau at steps 0/50/100/200:", taus)
m.modality_router.eval()
lg, ch, t = m._router_log_gate(logits.detach())
print("6. eval: argmax", ch.tolist() == logits.argmax(-1).tolist(), "hard gate", lg.tolist()[0], "tau", t)
m._action_stream_log_gate = None
print("7. no gate -> parent bool mask:", m._build_multistream_mot_attention_mask(lens, A, tpf, torch.device(dev), "all").dtype)
