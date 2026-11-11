"""RGB-only student [R0 | R_fut | A] distilled from a frozen multimodal teacher (action only).

Student = FastWAMJoint (exactly the RGB baseline). Teacher = any FastWAMJointMultimodal run
(MM+token, or the privileged FastWAMJointMultimodalPriv). Per training step:
  1. Student: usual noise draws (t_v, t_a, noisy RGB latents, noisy action), RGB video loss vs GT.
  2. Teacher (no grad) on the SAME noisy RGB latents, noisy action, t_v, t_a; depth/flow latents
     noised at the same t_v (own noise), frame 0 as the data gives it;
     context = cached instruction + proprio through the teacher's own proprio encoder.
  3. Losses, over valid (non-pad) action steps:
       action : w(t_a) * || v_S - target ||^2                    (v = action head output, [B, T, a])
                target = GT flow-matching target (distill_action_target=gt: exactly the RGB baseline's
                action loss) or the teacher's v_T (=teacher: output-level KD, the original behaviour)
       KD     : w(t_a) * || LN(h_S) - LN(h_T) ||^2  (mean over D) (h = last-layer action hidden
                state = input of the linear action head, [B, T, D]; LN = parameter-free layer norm)
  loss = [lambda_video * L_video + lambda_action * L_action]  +  distill_feature_weight * L_KD
         (the RGB baseline's total loss)                          (separate KD term)
  Defaults (teacher, 0) = output-level KD only. Feature-KD ablation: (gt, 1.0) = RGB baseline + feature KD.

The teacher is a plain attribute: not in state_dict / checkpoints / optimizer / DeepSpeed. It is
built from its run dir on the first training step, so eval / inference never loads it (plain
FastWAMJoint path).
Logged: loss_video, loss_action (lambda_action * L_action, same as the baseline's log), loss_kd (w * L_KD),
student_err_gt, teacher_err_gt (action MSE vs the flow-matching target, unweighted).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)


class FastWAMJointDistill(FastWAMJoint):
    def setup_distill(self, teacher_run: str, teacher_step: int, feature_weight: float = 0.0,
                      action_target: str = "teacher") -> None:
        # object.__setattr__: keep the teacher out of the nn.Module tree.
        object.__setattr__(self, "_teacher_spec", (str(teacher_run), int(teacher_step)))
        object.__setattr__(self, "_teacher", None)
        if action_target not in ("teacher", "gt"):
            raise ValueError(f"distill_action_target must be 'teacher' or 'gt', got {action_target!r}")
        self.distill_feature_weight = float(feature_weight)
        self.distill_action_target = action_target
        logger.info("FastWAMJointDistill teacher=%s @ step %d, action target %s, feature weight %g",
                    teacher_run, int(teacher_step), action_target, self.distill_feature_weight)

    @staticmethod
    def _capture_head_input(model, store: dict, key: str):
        """Forward hook on the linear action head: store its input (last-layer action hidden state)."""
        return model.action_expert.head.register_forward_hook(lambda m, inp, out: store.__setitem__(key, inp[0]))

    def _get_teacher(self):
        if self._teacher is None:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            run, step = self._teacher_spec
            mcfg = OmegaConf.load(f"{run}/config.yaml").model
            mcfg.load_text_encoder = False         # cached instruction context
            mcfg.skip_dit_load_from_pretrain = True  # weights come from the checkpoint
            teacher = instantiate(mcfg, model_dtype=self.torch_dtype, device=str(self.device))
            ckpt = f"{run}/checkpoints/weights/step_{step:06d}.pt"
            teacher.load_checkpoint(ckpt)
            teacher = teacher.to(self.device).eval().requires_grad_(False)
            object.__setattr__(self, "_teacher", teacher)
            logger.info("Distill teacher loaded: %s (%s)", type(teacher).__name__, ckpt)
        return self._teacher

    @torch.no_grad()
    def _teacher_action(self, sample, noisy_rgb, noisy_action, t_v, t_a, fuse, tiled) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher action velocity v_T [B, T, a] and last-layer action hidden state h_T [B, T, D] (float32)."""
        teacher = self._get_teacher()
        noisy = {"rgb": noisy_rgb}
        for name in teacher._enabled_streams():
            if name == "rgb":
                continue
            z, z0, _ = teacher._encode_stream_latents(sample[f"video_{name}"], tiled=tiled)
            zt = teacher.train_video_scheduler.add_noise(z, torch.randn_like(z), t_v)
            zt[:, :, 0:1] = z0  # clean frame 0, as in the teacher's training_loss: RGB frame 0 (MM teacher, data
            #                     rgb_condition_modalities [depth, flow]) or the real D0 / F0 (priv teacher, [])
            noisy[name] = zt
        shared = teacher._build_shared_context_action(
            sample=sample, batch_size=noisy_rgb.shape[0], num_frames=int(sample["video"].shape[2]),
        )
        store = {}
        hook = self._capture_head_input(teacher, store, "h")
        try:
            _, v = teacher._joint_denoise_multistream(
                noisy_latents=noisy,
                latents_action=noisy_action,
                timestep_video=t_v,
                timestep_action=t_a,
                context=shared["context"],
                context_mask=shared["context_mask"],
                fuse_vae_embedding_in_latents=fuse,
            )
        finally:
            hook.remove()
        return v.float(), store["h"].float()

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        x0, action = inputs["input_latents"], inputs["action"]
        batch_size = x0.shape[0]
        fuse = inputs["fuse_vae_embedding_in_latents"]

        noise_v = torch.randn_like(x0)
        t_v = self.train_video_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=x0.dtype)
        x_t = self.train_video_scheduler.add_noise(x0, noise_v, t_v)
        target_v = self.train_video_scheduler.training_target(x0, noise_v, t_v)
        if inputs["first_frame_latents"] is not None:
            x_t[:, :, 0:1] = inputs["first_frame_latents"]

        noise_a = torch.randn_like(action)
        t_a = self.train_action_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=action.dtype)
        a_t = self.train_action_scheduler.add_noise(action, noise_a, t_a)
        target_a = self.train_action_scheduler.training_target(action, noise_a, t_a)

        patch_t, patch_h, patch_w = (int(s) for s in self.video_expert.patch_size)
        lat_t, lat_h, lat_w = x_t.shape[-3:]
        tpf = (lat_h // patch_h) * (lat_w // patch_w)
        store = {}
        hook = self._capture_head_input(self, store, "h")
        try:
            pred_v, pred_a = self._joint_denoise_core(
                latents_video=x_t,
                latents_action=a_t,
                timestep_video=t_v,
                timestep_action=t_a,
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                attention_mask=self._build_mot_attention_mask(
                    video_seq_len=(lat_t // patch_t) * tpf, action_seq_len=a_t.shape[1],
                    video_tokens_per_frame=tpf, device=x_t.device,
                ),
                fuse_vae_embedding_in_latents=fuse,
                action_condition=action,
            )
        finally:
            hook.remove()
        h_student = store["h"]

        # RGB video loss vs GT (as FastWAM.training_loss)
        if inputs["first_frame_latents"] is not None:
            pred_v, target_v = pred_v[:, :, 1:], target_v[:, :, 1:]
        video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_v, target_video=target_v, image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=inputs["first_frame_latents"] is None,
        )
        loss_video = (video_per_sample * self.train_video_scheduler.training_weight(t_v).to(video_per_sample)).mean()

        # action loss (GT or teacher target) + optional separate feature-KD loss; the teacher runs every step
        v_teacher, h_teacher = self._teacher_action(sample, x_t, a_t, t_v, t_a, fuse, tiled)
        pad = inputs["action_is_pad"]
        valid = torch.ones(pred_a.shape[:2], device=pred_a.device) if pad is None else (~pad).float()
        denom = valid.sum(1).clamp(min=1.0)
        w_a = self.train_action_scheduler.training_weight(t_a)

        def per_sample(err):  # [B, T, a] -> [B]
            return (err.mean(-1) * valid).sum(1) / denom

        # action output term: GT flow-matching target (= the RGB baseline's action loss) or the teacher's velocity
        out_target = v_teacher if self.distill_action_target == "teacher" else target_a.float()
        gap = per_sample(F.mse_loss(pred_a.float(), out_target, reduction="none"))
        loss_action = (gap * w_a.to(gap)).mean()

        # student loss = the RGB baseline's total (video + action) + the separate KD term
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        fw = self.distill_feature_weight
        if fw > 0:  # feature KD: last-layer action hidden state
            d = h_student.shape[-1]
            feat_gap = per_sample(F.mse_loss(
                F.layer_norm(h_student.float(), (d,)), F.layer_norm(h_teacher, (d,)), reduction="none"))
            loss_kd = (feat_gap * w_a.to(feat_gap)).mean()
            loss_total = loss_total + fw * loss_kd

        with torch.no_grad():
            student_err = per_sample((pred_a.float() - target_a.float()) ** 2).mean()
            teacher_err = per_sample((v_teacher - target_a.float()) ** 2).mean()
        logs = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),   # vs GT or teacher (distill_action_target)
            "student_err_gt": float(student_err.item()),
            "teacher_err_gt": float(teacher_err.item()),
        }
        if fw > 0:
            logs["loss_kd"] = fw * float(loss_kd.detach().item())
        return loss_total, logs
