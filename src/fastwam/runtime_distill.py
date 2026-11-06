"""Hydra factories for the privileged MM teacher and the RGB distillation student.

Both reuse the existing factories in runtime.py and only swap the class (the subclasses add no
parameters), so every model argument is exactly that of the base factory.
"""

from __future__ import annotations


def create_fastwam_joint_multimodal_priv(**kwargs):
    """FastWAMJointMultimodalPriv: MM + clean D0/F0 tokens (args of create_fastwam_joint_multimodal)."""
    from .models.wan22.fastwam_joint_multimodal_priv import FastWAMJointMultimodalPriv
    from .runtime import create_fastwam_joint_multimodal

    model = create_fastwam_joint_multimodal(**kwargs)
    model.__class__ = FastWAMJointMultimodalPriv
    return model


def create_fastwam_joint_distill(teacher_run: str, teacher_step: int = 21700, **kwargs):
    """FastWAMJointDistill: RGB FastWAMJoint student + frozen teacher from ``teacher_run`` @ ``teacher_step``."""
    from .models.wan22.fastwam_joint_distill import FastWAMJointDistill
    from .runtime import create_fastwam_joint

    model = create_fastwam_joint(**kwargs)
    model.__class__ = FastWAMJointDistill
    model.setup_distill(teacher_run, teacher_step)
    return model
