# Modality router: analysis, tests and results (2026-10-05 to 2026-10-08)

This folder holds the analysis scripts, tests, logs and notes from the inline modality-router work. Run every script from the repo root, `/data/mgovind/FastWAM`, after `source scripts/env.sh`.

## Folder contents

| path | what |
|---|---|
| `scripts/router_plus_compare.py` | LIBERO-Plus: router vs every baseline. Paired McNemar tests; router picks per suite / category, recovered from the worker logs |
| `scripts/router_libero_compare.py` | LIBERO: the same, per episode, plus router picks per suite and per task |
| `scripts/plus_compare.py` | LIBERO-Plus comparison for the **offline** routers (`eval_router_libero_plus_new.py`) |
| `scripts/label_noise.py` | How noisy the action-error oracle labels are (per-seed agreement, cross-seed oracle ceiling, per-task) |
| `scripts/flow_check.py` | Why the offline routers never pick Flow (p(flow) distribution, cost of a wrong Flow pick) |
| `scripts/r0_parity.py` | R0 train/inference parity: full-window vs single-frame VAE encode feeds the router the same input |
| `scripts/router_summary.py` | Router metrics from a run's `output.log`, averaged over 250-step windows (`python ... [log] [window]`) |
| `scripts/eval_latency.py` | Closed-loop sec / replan of every LIBERO eval run, from the tqdm bars in the worker logs (includes sim; flags overlapping evals) → `outputs/eval_latency.json` |
| `scripts/m_context_compare.py` | Random-select + modality context (target video): LIBERO per suite, LIBERO-Plus per category, paired McNemar vs baselines |
| `router_picks_decoding.ipynb` | Router full-all: router probabilities and picks per replan; decoded future of every slot vs GT (what each slot really contains) → `outputs/router_picks_decoding/` |
| `scripts/watch_tmux.py` | Read-only watcher for a training run in tmux (errors, metrics, checkpoints) |
| `tests/smoke_router_new.py` | CPU: router pad masking, temperature, no gradient into proprio, timestep_ema baseline, sampling vs argmax |
| `tests/smoke_warmup.py` | CPU: `router_warmup_steps` (uniform picks, zero router gradient, resume clock, Gumbel warm-up) |
| `tests/smoke_gumbel*.py` | Gumbel / one-hot gate: mask topology, gated full = short, gradient through the real `flash_attention` |
| `tests/test_prompt_layout.py`, `tests/test_joint.py` | Real model: modality condition (separate / joint), full vs short layout losses (fp32: identical) |
| `tests/test_infer_full*.py` | Real model: full-layout inference (3 slots, one-hot, only the chosen future denoised) vs short |
| `tests/test_trainer_eval.py` | Real model: the trainer `evaluate()` call path (`prompt_text`, decode only the chosen future) |
| `tests/test_eval_prompt_parity.py` | CPU, no weights: the eval joint-prompt context equals the training one (cached T5 tokens, padding masked, proprio last) |
| `tests/test_per_task_map.py` | `EVALUATION.active_future_per_task` helper: 40 LIBERO tasks resolve, missing task raises, 1495 Plus variants → 40 base tasks (`PYTHONPATH=.`) |
| `tests/test_prompt_target.py` | Real model on CPU: which text context each expert receives for `modality_prompt_target: both / video` (training + inference) |
| `logs/` | Offline router runs (`run_*` = full features, `pooled_*` = pooled), feature compaction, real-model test logs |

## Code added or changed (all new options default off)

| file | change |
|---|---|
| `src/fastwam/models/wan22/modality_router_new.py` | Inline router: pads masked after the projection; runs in the parameter dtype (DeepSpeed bf16), returns fp32 logits |
| `src/fastwam/models/wan22/fastwam_joint_multimodal_select_new.py` | REINFORCE inline router. Features detached; reward `action` / `task`; baseline `batch` / `timestep_ema`; advantage normalisation; router loss only in train mode (router's own mode); `router_warmup_steps`; `router_layout: short / full` (one-hot action mask); `router_video_loss: chosen / all`; `modality_prompt` (`joint` / `separate`) and `modality_prompt_target` (`both` / `video`: action expert keeps the plain instruction); block-table MoT mask; inference uses the same layout (full: only the chosen future denoised); `prompt_text` for trainer eval; decodes only the denoised future; video timestep weight in fp32 (as in FastWAM) |
| `src/fastwam/models/wan22/fastwam_joint_multimodal_select_gumbel.py` | Straight-through Gumbel router (inherits the mask / gate code; rejects `modality_prompt`) |
| `src/fastwam/runtime_new.py` | Factories `create_fastwam_joint_multimodal_select_new` / `_gumbel` |
| `configs/model/fastwam_joint_multimodal_select_{new,gumbel}.yaml` | Model configs with all the new keys |
| `configs/task/libero_joint_2cam224_1e-4_router_{new,gumbel,new_full_prompt}` + `libero_joint_2cam224_1e-4_router_short_m_context.yaml` | Task configs |
| `scripts/precompute_modality_text_embeds.py` | Caches the T5 embeddings of the modality conditions into `data/text_embeds_cache/libero_modality` (123 files) |
| `notebooks/train_modality_router_new.py` | Offline router training on a compact feature cache (`--features full / pooled`, `--loss`, `--ablate`) |
| `experiments/libero/eval_router_libero_new.py`, `experiments/libero_plus/eval_router_libero_plus_new.py` | Closed-loop eval of the offline routers (switching between unimodal experts) |
| `experiments/libero/eval_infer.py` | Fix: the `_new` select models use their own `infer_action` (otherwise the router was bypassed) |
| `src/fastwam/trainer.py` | Router params in the optimizer; `evaluate()` passes `prompt_text`; rollout PSNR/SSIM count RGB rollouts only; logs `eval/rollout_rgb_frac` |

Environment: `source scripts/env.sh` is enough (updated 2026-10-09). It adds `/data/mgovind/npp_libs` (CUDA-12 NPP symlinks) when `.venv` has no NPP libs. The current torchcodec (0.4.0, reinstalled 2026-10-07) doesn't even need NPP: it uses the system FFmpeg 6.1.1, which decodes the AV1 RGB videos. Don't add the `ltm` lib dir for training: its FFmpeg 4.2.2 has no AV1 decoder, so torchcodec falls back to pyav.

## Key findings

### Offline oracle labels (action error) are mostly noise
- **Seeds rarely agree:** pairwise per-seed agreement on the best stream is 53–56% (chance 45%), and all three seeds agree only 35% of the time.
- **The oracle gain is an illusion:** the in-sample oracle shows +33–40% over always-Depth, but choosing the label on two seeds and scoring on the held-out third gives **about 0%** (test split −0.3%).
- **Per-task routing gains nothing either:** about −0.7%.
- The user decided not to use these labels as a reference.

### Offline routers (test set, gain vs always-Depth)
| variant | full inputs | pooled inputs |
|---|---|---|
| CE, R0 + context | +1.7% | +1.1% |
| CE, context only | +1.8% | **+2.2%** |
| CE, R0 only | +0.4% | +1.0% |
| expected loss | 0.0 (always Depth) | 0.0 |
| CE, class-weighted | −11.2% | −13.2% |

Flow is never picked: p(flow) is at most about 0.3 and never above p(depth).

### Closed loop, offline routers switching between unimodal experts (LIBERO-Plus, cap 5)
- **Pooled context-only router 70.9%, full CE router 70.2%.** For comparison: unimodal RGB 71.1, Depth 69.4, Flow 67.7, MM 76.3.
- **The harness is deterministic:** all-Depth tasks have exactly the same outcome as the Depth baseline (508/508 and 829/829).

### Inline REINFORCE router
1. **First run, no warm-up, collapse_kl 0.02:** collapsed onto Flow within about 300 steps (about 75–90% Flow, Depth about 6%, entropy about 0.1). Stopped.
2. **Warm-up run** (`runs/libero_joint_2cam224_1e-4_router_new/2026-10-06_21-44-40`): warm-up 2000, collapse_kl 0.1, timestep_ema. Usage stays balanced, but entropy drops to about 0.05 right after warm-up, so routing becomes a fixed per-task lookup.

### Baseline for the next comparison: router without modality context (step 21,700, seed 42)
**LIBERO** (`evaluate_results/libero/libero_joint_2cam224_1e-4_router_new/20261008_121313`; 31,514 replans)

| suite | picks R / D / F | success |
|---|---|---|
| spatial | 0 / 0 / 100 | 98.6 |
| object | 100 / 0 / 0 | 99.6 |
| goal | 0 / 0 / 100 | 98.4 |
| libero_10 | 19.3 / 59.9 / 20.8 | 95.4 |
| **all** | **29.5 / 25.5 / 44.9** | **98.00** |

- **libero_10 per task:** RGB → 0, 7; Depth → 1, 2 (96%), 3, 5, 8, 9; Flow → 4, 6. One stream per task for 39 of 40 tasks.
- **Same-seed baselines:** RGB 97.95, Depth 98.05, Flow 98.40, gt-rgb 97.45, gt-all 98.60. None of the differences are significant (saturated).

**LIBERO-Plus** (`evaluate_results/libero_plus/libero_joint_2cam224_1e-4_router_new/20261008_073051`; 4 categories, cap 5, 1495 tasks; 37,913 replans)
- **Picks R/D/F:** all 29.1 / 20.4 / 50.5. Per suite: spatial 0/1/99, object 96/2/2, goal 0/0/100, libero_10 18/61/21. Per category (almost identical): Camera 26/24/50, Background 22/27/50, Noise 28/26/47, Layout 32/17/51.
- **Success:** Camera 43.0, Background 65.5, Noise 97.0, Layout 68.3, **total 68.36%**.
- **Paired, same seed:** the router is significantly worse than random-select `rgb` (72.04), unimodal RGB (71.10) and MM (76.32). It is not significantly different from gt (68.6–69.7) or unimodal Depth/Flow.

### Architecture facts checked in the code
- **Shared weights:** all video streams share one video expert, with no per-stream weights or embeddings. Future slots use identical RoPE positions and symmetric masks.
- **Single-future models:** random / gt / router (`select`, `select_new` short) build only the chosen future. `select_onepass` is the existing 3-slot masked variant (never trained).
- **`active_future` without a condition:** at inference it only changes the noise seed, because nothing tells the model which modality to generate. This is the motivation for the modality condition.
- **Short layout = FastWAMJoint:** the short-layout MoT mask is identical to `FastWAMJoint`'s (action attends all video). `FastWAM` (non-joint) lets the action see only R0.
- **Full + `chosen` = short:** they give identical losses and actions (fp32 difference ≤ 3e-8). Full only costs more (718 vs 326 tokens per sample).

## Next run (launched 2026-10-08): short layout + modality context
- **Task:** `libero_joint_2cam224_1e-4_router_short_m_context`. The context is `T5("<instruction> Predict the <modality> video.") + proprio`; the router itself sees `[instruction | proprio]`. Warm-up 2000, collapse_kl 0.1, timestep_ema.
- **Compare (planned Sat 2026-10-10)** against the tables above:
  - Is routing still a per-task lookup?
  - Does the pick mix change across perturbation categories?
  - LIBERO-Plus success vs 68.36 (no context), 72.0 (random-select) and 76.3 (MM).
- **Commands:**
  ```bash
  bash scripts/train_zero1.sh 4 task=libero_joint_2cam224_1e-4_router_short_m_context
  R=runs/libero_joint_2cam224_1e-4_router_short_m_context/<run_dir>
  bash scripts/eval_libero_plus_3seeds.sh --task libero_joint_2cam224_1e-4_router_short_m_context \
      --ckpt $R/checkpoints/weights/step_021700.pt --stats $R/dataset_stats.json \
      --categories Noise,Layout,Background,Camera --max-tasks-per-category 5 --seeds 42
  bash scripts/eval_libero_3seeds.sh --task libero_joint_2cam224_1e-4_router_short_m_context \
      --ckpt $R/checkpoints/weights/step_021700.pt --stats $R/dataset_stats.json --seeds 42
  ```
- **Analysis:** point `runs` in `scripts/router_plus_compare.py` / `scripts/router_libero_compare.py` at the new eval folders.

## Control run: random selection + modality context (added 2026-10-09)
- **Task:** `libero_joint_2cam224_1e-4_random_short_m_context_video`. It is the router run `..._router_short_m_context_video` with `modality_select: random`: same short layout, same condition (joint, target = video), no learned router.
- **Eval:** `EVALUATION.active_future: random` picks a stream uniformly at each replan, and the condition follows the pick. A fixed `--active-future rgb|depth|flow` evaluates one stream.
- **Defaults:** with the condition on and no stream given (e.g. the trainer's periodic eval), the model defaults to rgb.
- **Use:** the learned router should beat this run's best fixed stream and its random-at-eval result; otherwise routing adds nothing.

## Modality-decoding check: context vs no context (2026-10-10)
Script: `scripts/check_modality_decoding.py`. For each of 9 windows spread over the dataset, the same input and seed are generated with `active_future` = rgb / depth / flow, and each output is compared with the GT rgb / depth / flow futures (10 denoising steps, CPU, fp32). Outputs: `outputs/modality_decoding/<run>_step<step>/` (`results.json`, `windowNN_*.png/mp4`; rows = generated rgb/depth/flow, then GT rgb/depth/flow).

| run | mean PSNR, matching GT (rgb / depth / flow) | requested modality produced |
|---|---|---|
| `router_short_m_context` step 10k (condition, target both) | 19.2 / 22.5 / 26.0; off-diagonal 3.7–6.6 | **9 / 9 windows**, every stream |
| `router_new` step 21.7k (no condition) | table flat: about 10–15 in every cell | **never controllable**: all 3 requests give the same modality |

- **The no-context model generates one fixed modality per window**, whatever is requested. The windows in suite order give flow, flow, rgb, rgb, flow, depth, depth, depth, depth. This matches the router's per-suite picks (spatial → Flow, object → RGB, goal → Flow, libero_10 → mostly Depth). The video expert learned to infer the modality from the task text, because the router always paired each task with one stream. The window → suite mapping is inferred from dataset order.
- **With the condition, the modality follows the prompt.**

## Eval latency from the logs (2026-10-10)
`scripts/eval_latency.py`: wall time per replan (10 env steps + one model call, 10 denoising steps, 2000 episodes per run). This is not pure model latency, because it includes the sim and any server load.

| model (clean, same period Aug 26 – Sep 1) | s / replan |
|---|---|
| joint unimodal RGB (5 runs) | 0.433–0.466 (median 0.437) |
| joint unimodal Flow (Aug 26 / 31) | 0.431–0.491 |
| MM `rgb_depth_flow` (6 runs, Aug 29, no overlap) | 0.512–0.525 (median 0.515) |

- **Same-period comparison:** MM is about **1.18×** unimodal RGB per replan, despite 718 vs 326 MoT tokens. The per-replan time includes the sim and rendering, so the model share is smaller than the total.
- **Later runs are confounded:** the same MM checkpoint ranges from 0.86 to 1.57 when evals overlapped (Aug 29–30). Other late runs are also slow: rgb_depth 0.96, modality_gt 0.63–0.95, router_new 1.18 (Oct 8, while training and CPU jobs ran).
- **Pure model latency needs a direct benchmark:** time `infer_action` with CUDA sync on an idle GPU.

## Eval fix: joint modality prompt padding (2026-10-10)
- **Bug:** in training, the joint prompt comes from the cache: raw T5 tokens with the padding **masked**. Closed-loop eval (`infer_action(prompt=...)`) re-encoded it with `encode_prompt`, which zeroes the padding and sets an **all-ones** mask, so the padding was visible. That is a train/eval mismatch, about 86–93 extra visible pad tokens in the video expert's cross-attention.
- **Not affected:** the plain instruction (action expert; baselines) uses zeroed, visible padding in both training and eval (the Wan2.2 convention), so it was already consistent.
- **Not exercised by the bug:** the decoding check and the trainer's periodic eval both used the cache path.
- **Fix:**
  - `_infer_modality_context` always uses the training function `_modality_prompt_context`.
  - `_load_joint_prompt` falls back to online T5 **with the real padding mask** when a text isn't cached (e.g. new LIBERO-Plus instructions).
- **Coverage:** all 120 LIBERO eval prompts (40 tasks × 3 streams) hit the cache.
- **Who needs it:** this applies to every closed-loop eval of the `m_context` runs (`router_short_m_context`, `random_short_m_context_video`, `router_short_m_context_video`). No evals of these runs existed before the fix.

### LIBERO-Plus check (2026-10-10)
- **Same pipeline:** `eval_libero_plus_single.py` passes the raw prompt to the same `call_action_infer`, so the fix applies there too.
- **Plus rewords instructions:** the BDDL `:language` differs from LIBERO, e.g. "pick **the akita** black bowl …".
- **Cache hits** in the 4-category, cap-5 set: spatial 0/377, object 0/395, goal 193/385, libero_10 338/338. That is 25 distinct new instructions, which take the online fallback (real padding mask, verified: same mask as the cache).
- **T5 bf16 depends on batch size:**
  - Batched GPU encoding reproduces the cache exactly.
  - Single-text encoding (what every eval does, baselines included) differs a little: most tokens have cosine > 0.99, with a few outlier tokens (up to 27% rel. error on one prompt).
  - This affects all models equally. It is not new.

## Results: random-select + modality context, target video (2026-10-11)
Run `runs/libero_joint_2cam224_1e-4_random_short_m_context_video/2026-10-10_09-57-20`, step 21,700, seed 42. Script: `scripts/m_context_compare.py`; log: `logs/m_context_compare_20261011.log`.

**LIBERO overall:** random 97.80 · rgb 97.55 · depth 97.60 · flow 97.40. All saturated, and similar to the earlier baselines (97.95–98.40).

**LIBERO-Plus** (Camera / Background / Noise / Layout / total):

| run | Camera | Background | Noise | Layout | total |
|---|---|---|---|---|---|
| ctx random | 47.0 | 77.0 | 96.0 | 69.3 | **70.90** |
| ctx rgb | 42.5 | 74.0 | 96.0 | 66.8 | 68.43 |
| ctx depth | 49.5 | 76.5 | 96.0 | 68.2 | 70.50 |
| ctx flow | 45.0 | 75.0 | 95.0 | 69.8 | 70.57 |
| MM | 64.5 | 79.5 | 97.5 | 73.5 | **76.32** |
| no-ctx random-select (rgb / depth / flow) | | | | | 72.0 / 72.2 / 72.4 |
| unimodal RGB / Depth / Flow | | | | | 71.1 / 69.4 / 67.7 |
| no-ctx router | | | | | 68.36 |

**Paired comparisons:**
- **vs MM:** every context run is significantly worse (p < 1e-7).
- **vs the no-context router:** ctx random is better (p = 0.013).
- **vs no-ctx random-select rgb:** ctx rgb is significantly worse (p = 1e-4). ctx random and ctx depth are not significantly different.
- **random vs fixed streams:** random at eval beats fixed rgb (p = 0.003). It is equal to fixed depth.

**Takeaways:**
- The modality condition gives real per-stream control (decoding check), but it does **not** improve robustness. Single-stream ceiling ≈ 70.5–70.9, below no-ctx random-select (≈ 72) and MM (76.3).
- A router choosing among these streams can at best match the best stream, which is about 70.5 here, unless the choice varies per perturbation.
- The context model is strong on Background (74–77 vs unimodal RGB 57.5) and weak on Camera (42–50 vs MM 64.5 / unimodal RGB 61.5).
- Caveats: one seed, one trial per task.

### Implementation check before concluding (2026-10-11)
Script `scripts/verify_m_context.py` (log `logs/verify_m_context.log`); decoding check `logs/modality_decoding_random_video_step21700.log`.

1. **Eval loaded the right model:** `step_021700.pt` and the run's own `dataset_stats.json`, no missing-key warnings. The eval code is post-fix. Uncached Plus prompts used the online fallback.
2. **The condition works on the final checkpoint:** "Predict X" generates X in **9/9** windows for every stream. Matching-GT PSNR: RGB 22.0, Depth 25.9, Flow 29.8; off-diagonal 2–7. This also shows the training prompt ↔ target-stream alignment is correct.
3. **Train/eval parity, end to end:** `infer_action` via the eval path (prompt string) vs the training-style path (dataset context + cached joint prompt), same seed. Max action difference about 0.008–0.02 (bf16), identical MSE for rgb, depth and flow.
4. **Open-loop action MSE** (24 dataset windows, same seed; all / arm / gripper):

| model | all | arm | gripper |
|---|---|---|---|
| ctx rgb | 0.0100 | 0.0030 | 0.052 |
| ctx depth | 0.0099 | 0.0028 | 0.052 |
| ctx flow | 0.0098 | 0.0028 | 0.052 |
| unimodal RGB | 0.0090 | 0.0027 | 0.047 |
| MM | 0.0071 | 0.0025 | 0.035 |

The context model is in the unimodal range; there is no sign of a broken path. MM's edge is mostly the gripper.

**Not covered:** sim observation preprocessing (shared with every baseline) and multi-seed variance.

### Future-video quality (2026-10-11)
**RGB future** (`scripts/rgb_future_quality.py`): 24 dataset windows, same input, seed 42, 10 steps; PSNR/SSIM of decoded frames 1..T vs GT RGB.

| model | PSNR | SSIM |
|---|---|---|
| ctx model, `active_future=rgb` | 23.22 | 0.846 |
| unimodal RGB | **24.66** | 0.866 |
| MM, "rgb" slot | 8.68 | 0.351 |

- **ctx vs unimodal:** the ctx model is 1.4 dB worse and better in only 5/24 windows. It trains RGB on about 1/3 of samples.
- **MM slots have no identity** (`scripts/mm_slot_identity.py`):
  - The "rgb" slot generates **depth** in 5 of 6 windows (≈ 30 dB vs GT depth, ≈ 6 dB vs GT rgb).
  - The "flow" slot generates depth in 1 of 6.
  - The slots share weights and RoPE and have no stream embedding, so in free generation the modality of a slot is not controlled.
  - MM's 76.3 on Plus therefore comes with mostly depth-like futures, not with three distinct modalities. This supports adding stream embeddings to MM.

**Per stream: context model vs the unimodal model of that stream** (`scripts/stream_future_quality.py --stream depth|flow`; RGB from `rgb_future_quality.py`). 24 windows, seed 42, 10 steps.

| stream | ctx model PSNR / SSIM | unimodal PSNR / SSIM | ctx − unimodal | ctx better in |
|---|---|---|---|---|
| RGB | 23.22 / 0.846 | 24.66 / 0.866 | −1.44 dB | 5/24 |
| Depth | 26.30 / 0.922 | 28.11 / 0.938 | −1.81 dB | 3/24 |
| Flow | 29.07 / 0.963 | 28.65 / 0.963 | +0.42 dB | 13/24 |

- The shared, conditioned model is slightly worse than a dedicated model on RGB and depth, and on par on flow.
- Flow PSNR is inflated by the large near-zero (white) regions.
- Unimodal flow checkpoint: step 21000, the one used in its evals.

### Stream oracles: ctx model vs unimodal models (2026-10-11)
`scripts/oracle_compare.py`, log `logs/oracle_compare_20261011.log`. Seed 42; each oracle is computed the same way for the 3 conditioned streams of the ctx model and for the 3 unimodal models.

| oracle | LIBERO ctx | LIBERO unimodal | Plus ctx | Plus unimodal |
|---|---|---|---|---|
| single best fixed | 97.60 | 98.40 | 70.57 | 71.10 |
| per-episode OR (hindsight; 3 tries) | 99.65 | 99.75 | 77.73 | **82.34** |
| per-task best (in-sample) | 98.75 | 99.10 | 73.38 | 74.78 |
| held-out per-task (2-fold over variants, 40 base tasks) | – | – | 70.57 | 71.30 |

References on Plus: ctx random 70.90, MM 76.32.

- **Held-out, a router gains nothing:** a realistic per-task router (held-out) equals the best fixed stream for both groups. The in-sample per-task gain (+2.8 / +3.7) does not survive held-out selection, the same pattern as the offline action-error labels.
- **The unimodal models complement each other more** (per-episode OR 82.3 vs 77.7). Separate models make more independent errors than the 3 conditioned streams of one shared model.
- **MM (76.3) beats every non-hindsight oracle,** so combining streams beats selecting one.

### ctx model oracles by LIBERO-Plus perturbation category (2026-10-11)

| category (n) | rgb | depth | flow | random/replan | per-category (1 stream) | per task × category (hindsight) | held-out | per-episode OR |
|---|---|---|---|---|---|---|---|---|
| Camera (200) | 42.5 | **49.5** | 45.0 | 47.0 | 49.5 (depth) | 60.5 | 49.5 | 66.0 |
| Background (200) | 74.0 | 76.5 | 75.0 | 77.0 | 76.5 (depth) | 83.5 | 77.5 | 84.0 |
| Noise (200) | 96.0 | 96.0 | 95.0 | 96.0 | 96.0 | 97.5 | 96.0 | 97.5 |
| Layout (895) | 66.8 | 68.2 | **69.8** | 69.3 | 69.8 (flow) | 71.5 | 67.5 | 74.5 |
| **all (1495)** | 68.4 | 70.5 | 70.6 | 70.9 | **71.5** | 75.1 | 70.2 | 77.7 |

- **Per-category routing is the only real, non-hindsight gain:** +0.9 over the best single stream (71.5 vs 70.6), with depth for Camera and Background and flow for Layout. A router would have to recognise the perturbation type from the image.
- **Per (task × category) routing does not hold up:** held out, it falls to 70.2.
- **Camera has the largest headroom** (hindsight 60.5–66 vs 49.5) but no stable per-task signal.

### Per-task stream map in eval (2026-10-11)
`EVALUATION.active_future_per_task=<map.json>` (flag `--active-future-per-task` in both `*_3seeds.sh`).
- **Format:** `{suite: {task_id | base_task: rgb|depth|flow}}`. It sets `active_future` per task and overrides a global `active_future`.
- **Missing task:** a task not in the map raises an error.
- **LIBERO-Plus keys:** the base task name (variant suffix stripped by `plus_base_task`) or the task id.
- **Results:** `active_future` is recorded in each task's `*_results.json`.
- **LIBERO map** (from seed-42 fixed streams, ties → depth > rgb > flow): `outputs/per_task_stream_map_libero_ctx.json`. Depth 29 tasks, RGB 8, flow 3; 98.75 at seed 42 by construction.
- **Smoke test** (libero_spatial, seed 43, 1 trial): every task ran its mapped stream; 10/10 success (`evaluate_results/smoke/per_task_map/`).
- **Test plan:** map vs fixed depth at seeds 43 and 44.
- **Top-2 variant** (2026-10-11): map values can be lists. `["depth","rgb"]` becomes `active_future: "random:depth,rgb"`, a uniform draw per replan among the listed streams (`_resolve_infer_active_future` in the `_new` model).
  - **Map:** `outputs/per_task_stream_map_libero_ctx_top2.json`, the two best seed-42 streams per task (ties depth > rgb > flow). Pairs: depth+rgb 27, flow+rgb 9, depth+flow 4. Expected at seed 42 if mixed 50/50: 98.2.
  - **Smoke test** (spatial, seed 43, 1 trial): every task drew only from its two streams; 10/10 success.

### Latency, dedicated 1-trial runs (2026-10-11)
Setup: 40 LIBERO tasks × 1 trial, seed 42, 4 GPUs, run one after the other on idle GPUs (`evaluate_results/latency/`). Analysis: `scripts/eval_latency.py --root evaluate_results/latency`.

| model | s / replan (mean / median) | first episode per worker | task wall time |
|---|---|---|---|
| unimodal RGB (compiled) | 0.437 / 0.444 | 0.40–0.46 | 10.0 s |
| MM, attend all (uncompiled) | 0.513 / 0.500 | 0.50–0.54 | 10.8 s |

- **MM ≈ 1.17×** unimodal RGB per replan, the same as the earlier clean runs (0.437 vs 0.515).
- No compile warm-up shows in the first episodes.
- These timings include 10 sim steps per replan, so the model-only ratio is larger.

**Denoising steps and replan length** (same 1-trial setup):

| model | denoise steps | replan steps | s / replan | success (40 ep) |
|---|---|---|---|---|
| unimodal RGB | 10 | 10 | 0.436 | 39/40 |
| unimodal RGB | 5 | 10 | **0.276** | 40/40 |
| MM | 10 | 10 | 0.510 | 40/40 |
| MM | 10 | 5 | 0.472 | 40/40 |
| MM | 5 | 10 | **0.322** | 40/40 |

- **Cost per denoising step** (from 10 vs 5 steps): RGB ≈ 32 ms, MM ≈ 38 ms (1.17×), although MM has 2.2× the tokens. At batch 1 the GPU is latency-bound.
- **Fixed per-replan cost** (VAE encode, sim, rendering, …): RGB ≈ 0.12 s, MM ≈ 0.13 s.
- **Sim cost:** replan 5 vs 10 gives about 7.6 ms per sim step, so ≈ 0.08 s of each 10-step replan.
- **MM at 5 steps (0.322) is faster than RGB at 10 steps (0.436).**
- **Success at 1 trial per task is not evidence** for "same success". Confirm with full 50-trial LIBERO and the Plus set.

## Results: router + modality context, target video (2026-10-12)
Run `runs/libero_joint_2cam224_1e-4_router_short_m_context_video/2026-10-10_03-23-54`, step 21,700, seed 42. LIBERO `20261011_234209`, Plus `20261011_234324`.

| | LIBERO | Plus | Plus Camera | Plus Background | Plus Layout | Plus Noise |
|---|---|---|---|---|---|---|
| **ctx router** | **98.15** | **71.04** | 51.0 | 71.0 | 69.7 | 97.0 |
| ctx random per replan | 97.80 | 70.90 | 47.0 | 77.0 | 69.3 | 96.0 |
| ctx best fixed | 97.60 (depth) | 70.57 (flow) | | | | |
| no-ctx router | 98.00 | 68.36 | | | | |
| MM | 98.25 | 76.32 | 64.5 | 79.5 | 73.5 | 97.5 |

**Paired comparisons:**
- **LIBERO:** router vs ctx random p = 0.46; vs fixed streams p = 0.11–0.25.
- **Plus:** vs random p = 0.95; vs depth / flow p = 0.64 / 0.68; vs fixed rgb **better** (p = 0.010).

**Picks are a per-task lookup on the text again:**
- **LIBERO:** 38/40 tasks use one stream for all 50 trials.
- **Plus:** 37/40 base tasks use one stream for every variant. The pick mix is flat across perturbation categories (~1/3 each).
- **The picks follow the wording, not the scene.** Spatial gets flow on LIBERO (9/10 tasks) but RGB 79% on Plus, where the spatial instructions are reworded ("pick the akita …").

**Takeaway:**
- The condition fixed the old router's loss: 71.0 vs 68.4.
- The router is no better than random per replan or the best fixed stream, and still below MM.

## Distillation from 3 unimodal teachers (2026-10-12)
- **Model:** `src/fastwam/models/wan22/fastwam_joint_multimodal_select_distill.py` (subclass of the `_new` context model; `_new` is unchanged).
- **Factory:** `src/fastwam/runtime_distill.py`. **Task:** `configs/task/libero_joint_2cam224_1e-4_distill_uni3_best_m_context_video.yaml`.
- **Student:** the random-select context model (short layout, joint prompt, target video).
- **Teachers:** the frozen unimodal RGB (21700), Depth (21700) and Flow (21000) joint models, run on the student's own noisy action, timesteps and noisy future latents (no grad).
- **Target:** per sample, the teacher closest to the flow-matching action target (`distill_select: best`; `mean` also available).
- **Loss:** student loss + `distill_lambda` · w(t_a) · ‖v_student − v_target‖².
- **Teachers stay out of the student:** they are plain attributes, built lazily at the first training step, and never in the state_dict, checkpoints, optimizer or DeepSpeed. Eval never loads them.
- **Train/eval mode:** taken from the DiT, because the trainer keeps the top-level model in eval() during training.
- **Test** `tests/test_distill.py` (CPU, real weights, all pass):
  - teachers are outside the student;
  - each sample gets its own prediction and best teacher;
  - the distillation gradient reaches only the student;
  - λ = 0 gives exactly the context-model loss;
  - distillation is off when the DiT is in eval mode.
  - Teacher errors (0.005–0.006) are in the range of the student's action loss (0.006).
- **Logged:** `loss_distill`, `distill_gap`, `distill_err_<m>`, `distill_best_<m>`.

## Results: random full-all (video loss on all futures, action reads 1 random future) (2026-10-14)
Run `runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37`, step 21,700, seed 42, random stream per replan. LIBERO `20261014_000925`, Plus `20261014_000754`.

| | LIBERO | Plus | Camera | Background | Layout | Noise |
|---|---|---|---|---|---|---|
| **full-all (1 future read, 3 supervised)** | **98.95** | **77.26** | 57.0 | **84.5** | **75.4** | **98.5** |
| MM (3 read, 3 supervised) | 98.25 | 76.32 | **64.5** | 79.5 | 73.5 | 97.5 |
| RGB+Flow bimodal (2 read, 2 supervised) | – | 74.18 | 57.5 | 74.0 | 72.8 | 97.0 |
| no-ctx random-select (1 read, 1 supervised; fixed rgb/depth/flow) | – | 72.0–72.4 | 46.5–51.5 | 72–73.5 | 71.4–72.0 | 96–98 |
| ctx random / ctx router (prompt) | 97.80 / 98.15 | 70.90 / 71.04 | | | | |
| unimodal RGB | 97.95 | 71.10 | 61.5 | 57.5 | 70.6 | 96.5 |

**Paired comparisons, Plus:**
- **vs MM:** +0.9 (p = 0.34), not different.
- **vs no-ctx random-select:** +4.9 to +5.2 (p < 1e-6).
- **vs RGB+Flow:** +3.1 (p = 0.001).
- **vs the context models:** about +6.3 (p < 1e-8).

**Paired comparisons, LIBERO:**
- **vs MM:** +0.7 (p = 0.06).
- **vs unimodal RGB and Depth:** +1.0 and +0.9 (p = 0.01 and 0.02).

**Takeaways:**
- **MM's gain comes from the video supervision** (all three futures trained every step), not from the action reading several futures. With one future read, full-all matches MM.
- **Camera is the exception:** it stays below MM (57 vs 64.5). Background, Layout and Noise are above MM.
- **Inference cost can match a single-stream model:** futures never attend to each other and the action reads only the chosen one, so the unchosen futures can be dropped at inference without changing the result (full + chosen ≡ short). That gives single-stream cost (326 tokens instead of 718).

## Results: router full-all vs random full-all (2026-10-14, partial on Plus)
Run `runs/libero_joint_2cam224_1e-4_router_full_all/2026-10-12_22-29-10`, step 21,700, seed 42. LIBERO `20261014_184947` (full layout). Plus `20261014_195653`, 1,337/1,495 tasks at the time of writing.

**LIBERO:**
- **Success:** router 98.55 vs random 98.95 (p = 0.30) vs MM 98.25 (p = 0.48), so no difference.
- **Picks R/D/F:** 24 / 15 / 61; 39/40 tasks use one stream (per-task lookup).
- **Short vs full layout,** router, 33 tasks so far: identical success (12/12 discordant).

**LIBERO-Plus**, on the 1,337 common tasks:

| | Camera | Background | Layout | Noise | total |
|---|---|---|---|---|---|
| router full-all | **67.5** | 86.0 | 75.9 | 97.5 | **79.36** |
| random full-all | 57.0 | 84.5 | 77.5 | 98.5 | 78.61 |
| MM | 64.5 | 79.5 | 75.3 | 97.5 | 77.64 |

- **Total:** router vs random p = 0.46; vs MM p = 0.11.
- **Camera:** router vs random **+10.5, p = 0.010** (41/20 discordant); vs MM p = 0.57.
- **Background:** router vs MM +6.5, p = 0.024.
- **Layout:** router vs random −1.6, p = 0.15.
- **Picks R/D/F:** 54 / 13 / 33. Per category: Camera 50/14/36, Background 45/11/44, Layout 60/13/27, Noise 42/16/42. 31/40 base tasks use one stream for all variants (the prompt router had 37/40).
- **Random full-all, short vs full layout (complete Plus):** 77.79 vs 77.26, p = 0.49, so equivalent.

**To check next:** fixed-stream Plus evals of random full-all (rgb / depth / flow). They show whether the Camera gain is just "the RGB stream is better under camera shift", which the router picks more often.

### Final LIBERO-Plus (complete, 1,495 tasks)
| | Camera | Background | Layout | Noise | total |
|---|---|---|---|---|---|
| **router full-all** | **67.5** | **86.0** | 74.3 | 97.5 | **78.06** |
| random full-all (full / short layout) | 57.0 / 60.5 | 84.5 / 84.5 | 75.4 / 75.5 | 98.5 / 98.5 | 77.26 / 77.79 |
| MM | 64.5 | 79.5 | 73.5 | 97.5 | 76.32 |
| RGB+Flow bimodal | 57.5 | 74.0 | 72.8 | 97.0 | 74.18 |
| unimodal RGB | 61.5 | 57.5 | 70.6 | 96.5 | 71.10 |

**Router vs random:**
- **Total:** +0.8, p = 0.39.
- **Camera:** **+10.5, p = 0.010.** Random short got 60.5 on Camera, so bf16 noise alone moves Camera by about 3.5; the gain is still about +7.
- **Layout:** −1.1, p = 0.30.

**Router vs other models:**
- **vs MM:** total +1.7 (p = 0.08); Background +6.5 (p = 0.024).
- **vs RGB+Flow:** +3.9 (p = 2e-5).
- **vs unimodal RGB:** +7.0 (p = 8e-11).

**Picks R/D/F:** 49 / 12 / 40. Categories: Camera 50/14/36, Background 45/11/44, Layout 50/11/39, Noise 42/16/42. 31/40 base tasks use one stream for all variants; 236/1495 variants switch streams within the episode.

## Complete LIBERO-Plus (all 7 categories, 10,030 tasks, seed 42) (2026-10-15)
Router full-all with the short layout at inference: `evaluate_results/libero_plus_full/libero_joint_2cam224_1e-4_router_full_all_short/20261014_232043`.

| | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|
| **router full-all (short)** | **59.3** | 83.2 | **93.8** | **98.8** | **88.4** | 78.6 | 82.3 | **82.5** |
| MM | 52.5 | 82.9 | 91.7 | 95.4 | 80.6 | 78.0 | 83.1 | 79.8 |
| tasks | 1,599 | 1,550 | 1,537 | 1,142 | 1,076 | 1,601 | 1,525 | 10,030 |

**Paired tests:** on the 4 categories with an MM run on disk (5,801 tasks: `rgb_depth_flow/20260901_215136`).
- **vs MM:** +3.3, p = 7e-11. Camera **+6.9** (p = 8e-8), Background **+7.8** (p = 4e-15), Layout −0.8 (p = 0.33), Noise +0.7 (p = 0.49).
- **vs unimodal RGB / Depth / Flow:** +8.7 / +11.4 / +17.8 (all p < 1e-40).
- **Robot / Language / Light vs MM:** +0.3 / +2.1 / +3.4. Not paired; MM's run for these categories isn't stored locally.

**Caveats:**
- One seed.
- No complete-benchmark run of random full-all yet, so the +2.7 over MM can't be split into "full-all training" vs "router".

## Router full-all: picks vs decoded slot content (2026-10-15)
Notebook `router_picks_decoding.ipynb` (seed 42, 10 steps, short layout).

**Slot content** (40 tasks, first window of each; best-matching GT by PSNR):

| slot | looks like RGB | looks like depth | looks like flow |
|---|---|---|---|
| rgb slot | 0 | **40** | 0 |
| depth slot | 0 | **40** | 0 |
| flow slot | **27** | 8 | 5 |

**Router picks** (40 tasks): flow slot 27 (its content is RGB-like in 19, depth-like in 5, flow in 3), rgb slot 10 (content depth-like), depth slot 3 (depth).

**Router probabilities are soft:** e.g. ep0 is about 0.31 / 0.23 / 0.46 at every replan, so the argmax is decided by small margins and barely changes along the episode.

**Episode 0 trajectory:** flow picked at every replan. The flow slot decodes RGB-like at frames 0–50 and real flow at frames 60–90.

**Interpretation:** without slot identity, the slot names don't match their content.
- The router effectively chooses between a **depth-like future** (rgb or depth slot) and the **"flow" slot, which mostly contains an RGB-like future**.
- So on LIBERO-Plus, "rgb-slot picks" (49%) mean depth content, and "flow-slot picks" (40%) mostly mean RGB-like content.

## Router full-all + stream embedding + embed router input (added 2026-10-17)
- **Task:** `libero_joint_2cam224_1e-4_router_full_all_streamemb_embed` (21,700 steps, from scratch). Router full-all plus a per-stream adaLN embedding (init randn·0.02) and `router_input: embed`: the router reads the video expert's own patch embedding of R0 (98 tokens, attention-pooled) and `text_embedding` of [T5 | proprio] (mean-pooled text + proprio), detached.
- **Slot identity fixed:** `slot_supervision_check.py` at σ = 1 (step 10k) puts every slot on its own stream (rgb 0.13, depth 0.08, flow 0.05 rel. MSE; random full-all matched the wrong stream). Full 10-step all-slot `evaluate()`: rgb/depth/flow slots decode RGB/depth/flow, rgb slot vs GT 24.9 dB (random full-all 9.9). Trainer eval `infer_psnr` at step 20k: 35.4.
- **LIBERO, seed 42, layout full (2,000 episodes):**

  | Model | Spatial | Object | Goal | 10 | Total |
  |---|---|---|---|---|---|
  | MM | 99.2 | 99.8 | 97.8 | 96.2 | 98.25 |
  | random full-all | 99.0 | 99.8 | 98.6 | 98.4 | 98.95 |
  | router full-all | 98.8 | 99.8 | 98.2 | 97.4 | 98.55 |
  | **router full-all + streamemb + embed** | 99.4 | 99.2 | 98.6 | 99.0 | **99.05** |

  McNemar vs MM p = 0.026 (31 vs 15 discordant); vs router full-all p = 0.14; vs random full-all p = 0.86.
- **Picks** (`scripts/libero_router_picks.py`, per replan): rgb 31.6 / depth 19.6 / flow 48.8 %. Within-task variation is new: mean per-task pick entropy 0.42 vs 0.017 for router full-all; tasks with one stream ≥ 95 %: 19/40 vs 39/40. Spatial still ≈ all flow (99.1 %); Object, Goal and 10 mix all three. The old router was a per-task lookup; this one switches streams within a task.
- **LIBERO-Plus, capped 1,495 tasks, seed 42, layout full** (`scripts/plus_streamemb_compare.py`):

  | Model | Camera | Background | Noise | Layout | Total |
  |---|---|---|---|---|---|
  | **router full-all + streamemb + embed** | 58.0 | 83.5 | 96.5 | 74.4 | **76.39** |
  | router full-all | 67.5 | 86.0 | 97.5 | 74.3 | 78.06 |
  | random full-all | 57.0 | 84.5 | 98.5 | 75.4 | 77.26 |
  | MM | 64.5 | 79.5 | 97.5 | 73.5 | 76.32 |
  | unimodal RGB | 61.5 | 57.5 | 96.5 | 70.6 | 71.10 |

  McNemar vs new: router full-all p = 0.071 (Camera alone p = 0.025, 23 vs 42), random full-all p = 0.41, MM p = 1.0, unimodal RGB p < 0.001.
- **Picks on Plus:** rgb 33.8 / depth 21.4 / flow 44.9 %, nearly the same mix in every category. Base tasks whose majority pick differs across their perturbed variants: 22/40 (router full-all: 2/40). The router now reacts to the scene, but that did not buy robustness.
- **Camera hypothesis checked and rejected:** the old router's Camera 67.5 is not explained by reading depth content. Random full-all with a fixed 'rgb' slot (which decodes depth) scores Camera 50.0, fixed 'depth' slot 55.0, random 57.0.

## Router + stream embedding + embed input + RGB observation history (added 2026-10-20)
- **Task:** `libero_joint_2cam224_1e-4_router_full_all_streamemb_embed_hist` (run 2026-10-18_16-38-33, 21,700 steps). Same as streamemb + embed, plus `image_history_frames: 4` / `router_history: true`: the router also reads one VAE latent of RGB history (t-16 .. t-4, lead-in t-20 dropped), attention-pooled. Eval feeds it from `RouterHistoryBuffer` (checked end-to-end on 1 episode: history `[3, 5, 224, 448]` at all 25 replans).
- **LIBERO, seed 42:** spatial 99.0 / object 100.0 / goal 97.8 / 10 98.2 = **98.75** (streamemb + embed 99.05, p = 0.43; MM 98.25, p = 0.21; random full-all 98.95, p = 0.64). No measurable gain from history.
- **Picks:** rgb 31.7 / depth 22.2 / flow 46.1 %; per-task pick entropy 0.60 (streamemb + embed 0.42); 10/40 tasks with one stream >= 95 % (19/40). LIBERO-10 shifts to depth (41.5 %). History makes routing more state-dependent, not more successful.
- LIBERO-Plus eval started 2026-10-20 (pending).

## MM + cascade (cross_stream_attention: cascade) and slot identity (added 2026-10-20)
- **LIBERO, seed 42:** MM + cascade 98.80 (spatial 99.4 / object 99.8 / goal 99.0 / 10 97.0) vs MM 98.25 (+33/-22, p = 0.18).
- **The cascade mask gives implicit slot identity.** `slot_supervision_check.py` at σ = 1 (step 21.7k): MM + cascade rgb→rgb 0.082, depth→depth 0.064, flow→flow 0.031; MM rgb→flow 0.60, depth→rgb 0.48, flow→flow 0.82 (vs depth 0.90, ambiguous). Both correct for σ ≤ 0.95. Trainer eval PSNR: cascade 26.9 / 33.0 dB vs ~15 for MM. Asymmetric attention (R_f → R0; D_f → +R_f; F_f → +R_f + D_f) distinguishes the slots without a stream embedding.
- **So MM vs MM + cascade changes identity and interaction together.** MM + cascade + stream embedding (training) checks whether explicit identity adds anything on top.
