"""R1-Lite server (experiments/r1lite/serve_fastwam_r1lite.py): router action history for cf + ahist models.

The real FastWAMR1LitePolicy with the R1 cf+ahist task (sim_robotwin.yaml + task override), synthetic requests
(random camera images); checkpoint loading is skipped (weights don't matter for the plumbing).
  1. chunk 1 of an episode: history all pad (episode start).
  2. later chunks: history = the first replan_steps NORMALIZED rows of the previous chunks (last K), front-padded.
  3. a new demo_id (and chunk_id == 1) resets the history.
  4. the router decides (no active_future passed); the recorded model input contains the history.
  5. --replan-steps 8: only 8 rows per chunk enter the history.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/tests/test_r1_server_action_history.py
"""
import base64
import io
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/data/mgovind/FastWAM"
sys.path.insert(0, f"{REPO}/experiments/r1lite")
import types  # noqa: E402
for _m in ("fastapi", "uvicorn"):  # web stack only (not in this env); the policy class does not use it
    if _m not in sys.modules:
        try:
            __import__(_m)
        except ImportError:
            stub = types.ModuleType(_m)
            stub.FastAPI = stub.HTTPException = stub.run = object
            sys.modules[_m] = stub
import serve_fastwam_r1lite as srv  # noqa: E402
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew  # noqa: E402

TASK = "r1_lite_joint_3cam_384_1e-4_router_full_all_streamemb_embed_tok_cf_ahist"
STATS = f"{REPO}/runs/r1_lite_joint_3cam_384_1e-4_router_full_all_streamemb_embed/2026-10-16_18-01-04/dataset_stats.json"
OBS = "/data/mgovind/claude_scratch/r1_server_test_obs"
FastWAMJointMultimodalSelectNew.load_checkpoint = lambda self, path, optimizer=None: None  # weights irrelevant here


def img(h, w, seed):
    a = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def request(policy, run, demo, chunk):
    with open(policy._prompt_file) as f:
        prompt = f.read()
    return srv.InferenceRequest(
        prompt=prompt, state=[0.0] * srv.STATE_DIM, head_image=img(640, 640, chunk),
        left_wrist_image=img(360, 640, chunk + 1), right_wrist_image=img(360, 640, chunk + 2),
        run_id=run, demo_id=demo, chunk_id=chunk,
    )


def make_policy(extra):
    sys.argv = ["serve", "--config", f"{REPO}/configs/sim_robotwin.yaml", "--override", f"task={TASK}",
                "--checkpoint", STATS, "--dataset-stats", STATS, "--observations-dir", OBS, "--seed", "0", *extra]
    policy = srv.FastWAMR1LitePolicy(srv.parse_args())
    seen = []
    orig = policy.model.infer_action

    @__import__("functools").wraps(orig)  # keep the signature (the server inspects it)
    def spy(**k):
        out = orig(**k)
        seen.append({"hist": k.get("action_history"), "pad": k.get("action_history_is_pad"),
                     "af": k.get("active_future"), "out": out["action"].detach().float().reshape(-1, 14).cpu(),
                     "pick": out.get("active_future")})
        return out

    policy.model.infer_action = spy
    return policy, seen


def main():
    import json, shutil
    shutil.rmtree(OBS, ignore_errors=True)
    with open(f"{REPO}/data/real-world-r1-lite/meta/tasks.jsonl") as f:
        prompt = json.loads(f.readline())["task"]  # has a cached text embedding
    with open(f"{OBS}_prompt.txt", "w") as f:
        f.write(prompt)
    policy, seen = make_policy([])
    policy._prompt_file = f"{OBS}_prompt.txt"
    k, replan = policy.action_history.k, policy.action_history.replan
    assert (k, replan) == (32, 32), (k, replan)

    for c in (1, 2, 3):
        policy.infer(request(policy, "run_1", "demo_1", c))
    s1, s2, s3 = seen
    assert s1["pad"].all() and not s1["hist"].abs().sum(), "chunk 1 must be all pad"
    assert s1["af"] is None and s1["pick"] in ("rgb", "depth", "flow")
    print(f"1. chunk 1: history all pad; router picked {s1['pick']}  OK")
    assert not s2["pad"].any() and torch.equal(s2["hist"], s1["out"][:32])
    assert torch.equal(s3["hist"], s2["out"][:32])
    print(f"2. chunks 2/3: history = previous chunk's normalized output (32 rows); picks {s2['pick']}, {s3['pick']}  OK")
    policy.infer(request(policy, "run_1", "demo_2", 1))
    assert seen[3]["pad"].all()
    print("3. new demo_id / chunk_id 1 -> history reset  OK")
    policy.recorder.pending.join()  # async writer
    rec = [torch.load(p, weights_only=False) for p in sorted(__import__("pathlib").Path(OBS).rglob("model_input.pt"))]
    has = [isinstance(r, dict) and "action_history" in r for r in rec]
    print(f"4. recorded model inputs with action_history: {sum(has)}/{len(rec)}", "OK" if rec and all(has) else "CHECK")

    del policy
    torch.cuda.empty_cache()
    policy, seen = make_policy(["--replan-steps", "8"])
    policy._prompt_file = f"{OBS}_prompt.txt"
    for c in (1, 2, 3):
        policy.infer(request(policy, "run_1", "demo_1", c))
    s3 = seen[2]
    assert s3["pad"][:16].all() and not s3["pad"][16:].any()
    assert torch.equal(s3["hist"][16:24], seen[0]["out"][:8]) and torch.equal(s3["hist"][24:], seen[1]["out"][:8])
    print("5. --replan-steps 8: 8 executed rows per chunk, front-padded  OK")
    print("ALL OK")


if __name__ == "__main__":
    main()
