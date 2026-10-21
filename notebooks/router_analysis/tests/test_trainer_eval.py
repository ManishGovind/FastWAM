import os, numpy as np, torch
DEV = __import__("os").environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # TEST_DEVICE=cpu to avoid GPUs
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
REPO = "/data/mgovind/FastWAM"; RUN = f"{REPO}/runs/libero_joint_2cam224_1e-4_router_new/2026-10-06_21-44-40"
os.chdir(REPO)
with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_short_m_context"])
model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=DEV)
model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
model = model.to(DEV).eval()
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
loop = {}
orig = model._run_multistream_infer_loop
def spy(**k):
    init = {n: t.clone() for n, t in k["latents_video"].items()}
    lv, la = orig(**k)
    loop["changed"] = {n: bool((lv[n][:, :, 1:] - init[n][:, :, 1:]).abs().max() > 0) for n in init}
    return lv, la
model._run_multistream_infer_loop = spy
import inspect
print("infer accepts prompt_text:", "prompt_text" in inspect.signature(model.infer).parameters)
for i in np.linspace(0, len(ds) - 1, 3).round().astype(int):
    s = ds[int(i)]
    for layout in ("short", "full"):
        model.router_layout = layout
        with torch.no_grad():  # exactly the trainer.evaluate() call
            pred = model.infer(prompt=None, input_image=s["video_rgb"][:, 0].to(DEV).to(torch.bfloat16), num_frames=int(s["video_rgb"].shape[1]),
                               action=s["action"].to(DEV), action_horizon=int(s["action"].shape[0]), proprio=s["proprio"][0].to(DEV),
                               context=s["context"].to(DEV), context_mask=s["context_mask"].to(DEV), text_cfg_scale=1.0, action_cfg_scale=1.0,
                               num_inference_steps=4, seed=42, tiled=False, prompt_text=s["prompt"])
        dec = [k for k in pred if k.startswith("video_") and pred[k] is not None]
        same = pred["video"] is pred.get("video_" + str(pred.get("active_future")))
        print(f"window {int(i):6d} {layout:5s}: chosen={pred.get('active_future')} | denoised {loop['changed']} | decoded {dec} | video==chosen {same}")
