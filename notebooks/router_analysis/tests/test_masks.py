"""MoT attention masks: _new model vs the original FastWAMJoint / MM masks (CPU, no weights).

Checks
  1. FastWAMJoint [R0|R_fut|A] (first_frame_causal) == _new short layout, rgb.
  2. MM [R0|R|D|F|A], every action_attend_mode == _new full layout (no gate).
  3. MM restricted to one future == _new short layout, depth / flow.
  4. Block structure, cell by cell: R0 -> R0 only; each future -> R0 + itself (never another
     future, never the action); action -> R0 + allowed futures + itself.
  5. The float gated mask (full layout, 1-hot and 2-hot gates) gives the same SDPA output as
     MM's bool mask with the equivalent action_attend_mode, in fp32 and bf16.
  6. MM cross_stream_attention none / cascade / full: exact allowed blocks; only future->future
     blocks change.

    PYTHONPATH=. python notebooks/router_analysis/tests/test_masks.py
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastwam.models.wan22.wan_video_dit as wvd
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import _GATE_OFF
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew as New

TPF, FUT, A = 98, 196, 32  # tokens per frame, future tokens per stream, action tokens
STREAMS = ("rgb", "depth", "flow")
MODES = ("all", "rgb", "depth", "flow", "rgb_depth", "rgb_flow", "depth_flow", "cond_only")


def stub(cls, dtype=torch.float32):
    m = object.__new__(cls)
    nn.Module.__init__(m)
    m.enabled_video_streams = STREAMS
    m.torch_dtype = dtype
    return m


def joint_mask():
    v2v_cls = [c for c in vars(wvd).values() if isinstance(c, type) and "build_video_to_video_mask" in vars(c)][0]
    vexp = types.SimpleNamespace(video_attention_mask_mode="first_frame_causal")
    vexp.build_video_to_video_mask = types.MethodType(v2v_cls.build_video_to_video_mask, vexp)
    j = object.__new__(FastWAMJoint)
    nn.Module.__init__(j)
    j.__dict__["video_expert"] = vexp
    return FastWAMJoint._build_mot_attention_mask(j, TPF + FUT, A, TPF, "cpu")


def check_blocks(mask, names, sizes, allow):
    edges = [0]
    for s in sizes:
        edges.append(edges[-1] + s)
    for i, r in enumerate(names):
        for j, c in enumerate(names):
            cell = mask[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            want = c in allow[r]
            assert bool(cell.all()) if want else not bool(cell.any()), f"{r} -> {c}: expected {want}"


def main():
    new, mm = stub(New), stub(FastWAMJointMultimodal)
    full_lens = [TPF + FUT, FUT, FUT]

    # 1. FastWAMJoint == _new short rgb
    a = joint_mask()
    b = new._build_multistream_mot_attention_mask(stream_seq_lens=[TPF + FUT], action_seq_len=A, video_tokens_per_frame=TPF,
                                                  device="cpu", action_attend_mode="all", enabled_streams=("rgb",))
    assert torch.equal(a, b)
    check_blocks(b, ["R0", "R_fut", "A"], [TPF, FUT, A], {"R0": {"R0"}, "R_fut": {"R0", "R_fut"}, "A": {"R0", "R_fut", "A"}})
    print("1. FastWAMJoint [R0|R_fut|A] == _new short rgb  OK")

    # 2. MM == _new full (no gate), every mode, + block structure
    fut_of = {"rgb": "R_fut", "depth": "D_fut", "flow": "F_fut"}
    sees = New._ACTION_SEES
    for mode in MODES:
        a = FastWAMJointMultimodal._build_multistream_mot_attention_mask(mm, stream_seq_lens=full_lens, action_seq_len=A,
                                                                         video_tokens_per_frame=TPF, device="cpu", action_attend_mode=mode)
        b = new._build_multistream_mot_attention_mask_base(stream_seq_lens=full_lens, action_seq_len=A,
                                                           video_tokens_per_frame=TPF, device="cpu", action_attend_mode=mode)
        assert torch.equal(a, b), mode
        allow = {"R0": {"R0"}, "R_fut": {"R0", "R_fut"}, "D_fut": {"R0", "D_fut"}, "F_fut": {"R0", "F_fut"},
                 "A": {"R0", "A", *(fut_of[s] for s in sees[mode])}}
        check_blocks(b, ["R0", "R_fut", "D_fut", "F_fut", "A"], [TPF, FUT, FUT, FUT, A], allow)
    print(f"2. MM [R0|R|D|F|A] == _new full, all {len(MODES)} attend modes + block structure  OK")

    # 3. MM restricted to one future == _new short depth / flow
    for s in ("depth", "flow"):
        a = FastWAMJointMultimodal._build_multistream_mot_attention_mask(mm, stream_seq_lens=[TPF, FUT], action_seq_len=A,
                                                                         video_tokens_per_frame=TPF, device="cpu",
                                                                         action_attend_mode="all", enabled_streams=("rgb", s))
        b = new._build_multistream_mot_attention_mask(stream_seq_lens=[TPF, FUT], action_seq_len=A, video_tokens_per_frame=TPF,
                                                      device="cpu", action_attend_mode=s, enabled_streams=("rgb", s))
        assert torch.equal(a, b), s
        check_blocks(b, ["R0", "X_fut", "A"], [TPF, FUT, A], {"R0": {"R0"}, "X_fut": {"R0", "X_fut"}, "A": {"R0", "X_fut", "A"}})
    print("3. MM single future == _new short depth / flow  OK")

    # 5. gated float mask == MM bool mask under SDPA (1-hot and 2-hot), fp32 + bf16
    gates = {"rgb": [0, _GATE_OFF, _GATE_OFF], "depth": [_GATE_OFF, 0, _GATE_OFF], "flow": [_GATE_OFF, _GATE_OFF, 0],
             "rgb_depth": [0, 0, _GATE_OFF], "rgb_flow": [0, _GATE_OFF, 0], "depth_flow": [_GATE_OFF, 0, 0],
             "all": [0, 0, 0]}
    S = sum(full_lens) + A
    for dt in (torch.float32, torch.bfloat16):
        n = stub(New, dt)
        with n._action_stream_gate(torch.tensor(list(gates.values()))):
            gm = n._build_multistream_mot_attention_mask(stream_seq_lens=full_lens, action_seq_len=A, video_tokens_per_frame=TPF,
                                                         device="cpu", action_attend_mode="all")
        torch.manual_seed(0)
        q, k, v = (torch.randn(len(gates), 4, S, 64, dtype=dt) for _ in range(3))
        out_g = F.scaled_dot_product_attention(q, k, v, attn_mask=gm[:, None])
        for b, mode in enumerate(gates):
            bm = FastWAMJointMultimodal._build_multistream_mot_attention_mask(mm, stream_seq_lens=full_lens, action_seq_len=A,
                                                                              video_tokens_per_frame=TPF, device="cpu", action_attend_mode=mode)
            out_b = F.scaled_dot_product_attention(q[b:b + 1], k[b:b + 1], v[b:b + 1], attn_mask=bm)
            assert torch.equal(out_g[b:b + 1], out_b), (dt, mode)
    print(f"5. gated float mask == MM bool mask under SDPA ({len(gates)} gates incl. 1-hot / 2-hot; fp32 + bf16)  OK")

    # 6. MM cross_stream_attention: none == original; cascade / full add only future -> future blocks
    names, sizes = ["R0", "rgb", "depth", "flow", "A"], [TPF, FUT, FUT, FUT, A]
    base = FastWAMJointMultimodal._build_multistream_mot_attention_mask(
        mm, stream_seq_lens=full_lens, action_seq_len=A, video_tokens_per_frame=TPF, device="cpu", action_attend_mode="all")
    extra = {"none": {}, "cascade": {"depth": {"rgb"}, "flow": {"rgb", "depth"}},
             "full": {"rgb": {"depth", "flow"}, "depth": {"rgb", "flow"}, "flow": {"rgb", "depth"}}}
    for mode, add in extra.items():
        c = stub(FastWAMJointMultimodal)
        c.cross_stream_attention = mode
        m = FastWAMJointMultimodal._build_multistream_mot_attention_mask(
            c, stream_seq_lens=full_lens, action_seq_len=A, video_tokens_per_frame=TPF, device="cpu", action_attend_mode="all")
        allow = {"R0": {"R0"}, "A": {"R0", "rgb", "depth", "flow", "A"}}
        allow.update({n: {"R0", n} | add.get(n, set()) for n in STREAMS})
        check_blocks(m, names, sizes, allow)
        if mode == "none":
            assert torch.equal(m, base)
        else:
            assert torch.equal(m[TPF:TPF + 3 * FUT, TPF:TPF + 3 * FUT] | base[TPF:TPF + 3 * FUT, TPF:TPF + 3 * FUT],
                               m[TPF:TPF + 3 * FUT, TPF:TPF + 3 * FUT])          # only adds
            outside = torch.ones_like(m); outside[TPF:TPF + 3 * FUT, TPF:TPF + 3 * FUT] = False
            assert torch.equal(m[outside], base[outside])                       # R0 / action / video->A unchanged
    print("6. MM cross_stream_attention: none == original MM; cascade adds D->R, F->R+D; full adds all "
          "future->future blocks; R0, action rows and video->A unchanged  OK")
    print("ALL MASK CHECKS PASSED")


if __name__ == "__main__":
    main()
