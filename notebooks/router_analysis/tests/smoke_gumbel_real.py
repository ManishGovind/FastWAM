import torch, torch.nn as nn
DEV = __import__("os").environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # TEST_DEVICE=cpu to avoid GPUs
from fastwam.models.wan22.wan_video_dit import flash_attention
from fastwam.models.wan22.fastwam_joint_multimodal_select_gumbel import FastWAMJointMultimodalSelectGumbel as G
torch.manual_seed(0)
m = G.__new__(G); nn.Module.__init__(m)
m.enabled_video_streams = ("rgb", "depth", "flow"); m.torch_dtype = torch.bfloat16
m.modality_router = nn.Linear(4, 3).to(DEV).train()
m.router_gumbel_tau_start, m.router_gumbel_tau_end, m.router_gumbel_tau_anneal_steps = 1.0, 0.5, 100
m._gumbel_step = 0; m._action_stream_log_gate = None
tpf, lens, A, B, H, D = 98, [882, 784, 784], 32, 4, 24, 128
S = sum(lens) + A
for hard in (True, False):
    m.router_gumbel_hard = hard
    logits = torch.randn(B, 3, device=DEV, requires_grad=True)
    log_gate, choices, tau = m._router_log_gate(logits)
    with m._action_stream_gate(log_gate):
        mask = m._build_multistream_mot_attention_mask(lens, A, tpf, torch.device(DEV), "all")
    q, k, v = (torch.randn(B, S, H * D, device=DEV, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    out = flash_attention(q, k, v, num_heads=H, ctx_mask=mask)
    out[:, sum(lens):].float().pow(2).mean().backward()
    print(f"hard={hard}: S={S} mask {tuple(mask.shape)} choices {choices.tolist()} | grad->logits finite "
          f"{bool(torch.isfinite(logits.grad).all())} nonzero {float(logits.grad.abs().sum()) > 0} | "
          f"q.grad finite {bool(torch.isfinite(q.grad).all())} | peak mem {(torch.cuda.max_memory_allocated() / 2**30 if DEV == 'cuda' else 0.0):.1f} GiB")
