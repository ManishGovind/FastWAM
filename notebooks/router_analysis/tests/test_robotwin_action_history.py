"""RoboTwin eval policy + action history (experiments/robotwin/fastwam_policy/deploy_policy.py), no simulator.

  1. history model: _infer_action_chunk passes the buffer's history to infer_action (all pad at the episode start,
     then the first replan_steps NORMALIZED rows of every predicted chunk, last K, front-padded); reset() clears it;
  2. model without router_action_history: infer_action gets no history kwargs (behaviour unchanged);
  3. RoboTwin data (configs ..._streamemb_tok_embed_ahist): sample["action_history"] [32, 14] == the first action of
     samples t-32 .. t-1, action window unchanged, front pad at the episode start.

The policy is built with a stub model (records the infer_action kwargs); that infer_action(action_history=...)
gives the router the training features is covered by test_router_action_history.py (4b).

    .venv-Robottwin/bin/python notebooks/router_analysis/tests/test_robotwin_action_history.py   # 1, 2
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python notebooks/router_analysis/tests/test_robotwin_action_history.py   # 1-3
"""
import glob
import os
import sys
import types
from collections import deque

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = "/data/mgovind/FastWAM"
sys.path[:0] = [REPO, f"{REPO}/src"]
os.chdir(REPO)
from experiments.robotwin.fastwam_policy import deploy_policy as dp  # noqa: E402

K, A, REPLAN, H = 32, 14, 24, 32


class StubModel:
    device, torch_dtype = "cpu", torch.float32

    def __init__(self, history: bool):
        self.router_action_history, self.router_action_history_steps = history, K
        self.action_expert = types.SimpleNamespace(action_dim=A)
        self.calls, self.n = [], 0

    def infer_action(self, prompt, input_image, action_horizon, proprio, negative_prompt, text_cfg_scale,
                     num_inference_steps, sigma_shift, seed, rand_device, tiled,
                     action_history=None, action_history_is_pad=None):
        self.calls.append({"action_history": action_history, "action_history_is_pad": action_history_is_pad})
        self.n += 1   # chunk n: row i = 100 n + i (normalized space)
        return {"action": (100.0 * self.n + torch.arange(H, dtype=torch.float32)).unsqueeze(-1).expand(H, A).clone()}


def make_policy(model):
    p = dp.WorldActionRobotWinPolicy.__new__(dp.WorldActionRobotWinPolicy)   # skip checkpoint / processor loading
    p.model, p.action_horizon, p.replan_steps = model, H, REPLAN
    p.num_inference_steps, p.sigma_shift, p.seed, p.text_cfg_scale = 2, None, 0, 1.0
    p.negative_prompt, p.rand_device, p.tiled, p.timing_enabled, p._num_video_frames = "", "cpu", False, False, 9
    p.pending_actions, p.episode_count, p.step_count = deque(), 0, 0
    p._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
    p._history_cfg = OmegaConf.create({"EVALUATION": {"replan_steps": p.replan_steps}})
    p.action_history = dp.ActionHistoryBuffer(p.model, p._history_cfg)
    p._build_robotwin_image_tensor = lambda obs: torch.zeros(1, 3, 384, 320)
    p._normalize_state = lambda s: torch.zeros(1, A)
    p._denormalize_action = lambda a: a.unsqueeze(0).numpy() if a.ndim == 2 else a.numpy()
    return p


OBS = {"joint_action": {"vector": np.zeros(A, dtype=np.float32)}}


def main():
    # 1. history model
    m = StubModel(history=True); p = make_policy(m)
    for _ in range(3):
        p._infer_action_chunk(OBS, "lift the pot")
    c0, c1, c2 = m.calls
    assert c0["action_history_is_pad"].tolist() == [1.0] * K and c0["action_history"].abs().sum() == 0
    assert c1["action_history_is_pad"].tolist() == [1.0] * (K - REPLAN) + [0.0] * REPLAN
    assert c1["action_history"][K - REPLAN:, 0].tolist() == [100.0 + i for i in range(REPLAN)]
    executed = [100.0 * n + i for n in (1, 2) for i in range(REPLAN)]
    assert c2["action_history_is_pad"].sum() == 0 and c2["action_history"][:, 0].tolist() == executed[-K:]
    p.reset(); p._infer_action_chunk(OBS, "lift the pot")
    assert m.calls[-1]["action_history_is_pad"].tolist() == [1.0] * K, "reset() must clear the history"
    print(f"1. history model: all pad at start, +{REPLAN} executed normalized rows per chunk, last {K}, reset clears  OK")

    # 2. model without router_action_history
    m = StubModel(history=False); p = make_policy(m)
    p._infer_action_chunk(OBS, "lift the pot"); p._infer_action_chunk(OBS, "lift the pot")
    assert all(c["action_history"] is None and c["action_history_is_pad"] is None for c in m.calls)
    print("2. model without router_action_history: no history kwargs (unchanged)  OK")

    # 3. RoboTwin data (needs the main .venv: hydra configs + LeRobot dataset)
    try:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
    except ImportError:
        print("3. skipped (no hydra in this venv)"); return
    task = "robotwin_joint_3cam_384_1e-4_clean_router_full_all_streamemb_tok_embed_ahist"

    def comp(*o):
        with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
            return compose(config_name="train", overrides=[f"task={task}", *o])
    stats = sorted(glob.glob("runs/robotwin_joint_3cam_384_1e-4_clean_router_full_all_streamemb_embed/*/dataset_stats.json"))
    if not stats:
        print("3. skipped (no RoboTwin dataset_stats.json)"); return
    ds = instantiate(comp().data.train, pretrained_norm_stats=stats[-1])
    ds0 = instantiate(comp("data.train.action_history_steps=0").data.train, pretrained_norm_stats=stats[-1])
    fr = ds.lerobot_dataset.episode_data_index["from"].tolist()
    i = fr[2] + 60
    s = ds._get(i)
    assert s["action_history"].shape == (K, A) and not s["action_history_is_pad"].any()
    assert torch.equal(s["action"], ds0._get(i)["action"]), "action window must not change"
    for k in (1, 7, K):
        assert torch.allclose(s["action_history"][K - k], ds0._get(i - k)["action"][0], atol=1e-6), k
    s4 = ds._get(fr[2] + 4)
    assert s4["action_history_is_pad"].tolist() == [True] * (K - 4) + [False] * 4 and s4["action_history"][: K - 4].abs().sum() == 0
    print(f"3. RoboTwin data: history [{K}, {A}] == first action of samples t-{K}..t-1; window unchanged; t=4 -> 4 real rows  OK")


if __name__ == "__main__":
    main()
    print("ALL ROBOTWIN ACTION HISTORY CHECKS PASSED")
