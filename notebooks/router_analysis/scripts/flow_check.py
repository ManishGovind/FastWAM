import sys, numpy as np, torch
sys.path.insert(0, "notebooks")
from train_modality_router_new import Features, MODELS
from train_modality_router import split_by_episode, _as_np
from fastwam.models.wan22.modality_router_new import ModalityRouter
D = "notebooks/outputs/modality_action_labels/"
L = D + "labels_epall_stride1_s42-43-44_grip0.2148"
d = torch.load(L + ".pt", map_location="cpu", weights_only=False)
label = _as_np(d["label"]); loss = _as_np(d["weighted_action_loss"])
_, _, te = split_by_episode(_as_np(d["episode"]), _as_np(d["suite"]), 0)
feats = torch.load(L + "_router_feats_compact.pt", map_location="cpu", weights_only=False)
te_idx = torch.as_tensor(np.where(te)[0], device="cuda")
for name in ("ce_none_none", "pooled2_ce_none_none"):
    ck = torch.load(D + f"routers_new_{name}.pt", map_location="cpu", weights_only=False)
    x = Features(feats, "cuda", "none", np.where(~te)[0], ck["ablate"], features=ck.get("features", "full"),
                 r0_pool=ck.get("r0_pool") or 2, latent_shape=tuple(ck.get("latent_shape", (48, 14, 28))))
    r = ModalityRouter(r0_dim=ck["r0_dim"], context_dim=ck["context_dim"], context_len=ck["context_len"],
                       num_classes=3).cuda().eval()
    r.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        p = torch.cat([r(x.batch(te_idx[i:i+512])).softmax(-1) for i in range(0, len(te_idx), 512)]).cpu().numpy()
    pf = p[:, 2]
    print(f"\n== {name}: p(flow) on test  mean {pf.mean():.3f}  p50 {np.median(pf):.3f}  p99 {np.quantile(pf,.99):.3f}  max {pf.max():.3f}")
    print(f"   windows where p(flow) > p(depth): {(pf > p[:,1]).sum()}  | mean p R/D/F = {np.round(p.mean(0),3).tolist()}")
    lab = label[te]
    print(f"   mean p(flow) on flow-labeled windows {pf[lab==2].mean():.3f} vs others {pf[lab!=2].mean():.3f}")
    del x; torch.cuda.empty_cache()
# How costly is picking Flow when it is not the label?
te_l = loss[te]; lab = label[te]
print(f"\nflow-labeled test windows: {(lab==2).sum()} ({(lab==2).mean():.1%})")
print(f"  on flow-labeled windows, flow err / depth err: {te_l[lab==2,2].mean()/te_l[lab==2,1].mean():.3f}")
print(f"  on other windows,        flow err / depth err: {te_l[lab!=2,2].mean()/te_l[lab!=2,1].mean():.3f}")
