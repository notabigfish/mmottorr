from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_log_map, se3_exp_map, se3_geodesic_loss
from .quaternion import transform_to_quat_trans, quat_trans_to_transform, quat_trans_geodesic_loss
from .dual_quaternion import transform_to_dual_quaternion, dual_quaternion_to_transform, dual_quaternion_loss
from .matrix_representation import transform_to_matrix_features, matrix_features_to_transform, matrix_representation_loss


def transform_to_representation(T: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "se3_log":
        return se3_log_map(T)
    if representation == "quaternion_translation":
        return transform_to_quat_trans(T)
    if representation == "dual_quaternion":
        return transform_to_dual_quaternion(T)
    if representation == "matrix":
        return transform_to_matrix_features(T, mode="3x4")
    if representation in {"centroid_bias", "random_motor", "pga_feature", "shuffled_pairs", "no_pair_context"}:
        raise NotImplementedError(representation)
    raise ValueError(representation)


def representation_to_transform(rep: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "se3_log":
        return se3_exp_map(rep)
    if representation == "quaternion_translation":
        return quat_trans_to_transform(rep)
    if representation == "dual_quaternion":
        return dual_quaternion_to_transform(rep)
    if representation == "matrix":
        return matrix_features_to_transform(rep, mode="3x4")
    raise NotImplementedError(representation)


def representation_dim(representation: str, matrix_mode: str = "3x4") -> int:
    if representation == "se3_log": return 6
    if representation == "quaternion_translation": return 7
    if representation == "dual_quaternion": return 8
    if representation == "matrix": return 12 if matrix_mode == "3x4" else (16 if matrix_mode == "4x4" else 9)
    if representation == "centroid_bias": return 10
    if representation == "random_motor": return 6
    if representation == "shuffled_pairs": return 6
    if representation == "no_pair_context": return 6
    if representation in {"pga_feature", "pga_sandwich", "motordock_pga"}: return 8
    raise ValueError(representation)


def representation_loss(pred, target, representation: str, **kwargs) -> torch.Tensor:
    if representation in {"se3_log", "shuffled_pairs", "no_pair_context", "random_motor"}:
        return se3_geodesic_loss(se3_exp_map(pred), se3_exp_map(target), **kwargs)
    if representation == "quaternion_translation":
        return quat_trans_geodesic_loss(pred, target, **kwargs)
    if representation == "dual_quaternion":
        return dual_quaternion_loss(pred, target, **kwargs)
    if representation == "matrix":
        return matrix_representation_loss(pred, target, mode=kwargs.pop("matrix_mode", "3x4"), **kwargs)
    if representation in {"centroid_bias", "pga_feature"}:
        return pred.sum() * 0.0
    raise ValueError(representation)
