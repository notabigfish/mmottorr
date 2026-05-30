from __future__ import annotations

from dataclasses import dataclass
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim
from motordock.geometry.representation_conversions import representation_dim


@dataclass
class RepresentationSpec:
    name: str
    pair_feature_dim: int
    predicted_dim: int
    can_decode_to_transform: bool
    uses_transform_loss: bool
    description: str


def get_representation_spec(representation: str, matrix_mode: str = "3x4") -> RepresentationSpec:
    can_decode = representation in {"se3_log", "quaternion_translation", "dual_quaternion", "matrix", "shuffled_pairs", "no_pair_context"}
    uses_loss = representation not in {"centroid_bias", "no_pair_context"}
    pred_name = "se3_log" if representation in {"shuffled_pairs", "no_pair_context"} else representation
    if representation in {"pga_sandwich", "motordock_pga"}:
        pred_name = "pga_feature"
    return RepresentationSpec(
        name=representation,
        pair_feature_dim=representation_pair_feature_dim(representation, matrix_mode),
        predicted_dim=representation_dim(pred_name, matrix_mode),
        can_decode_to_transform=can_decode,
        uses_transform_loss=uses_loss,
        description=(
            "passive PGA feature only"
            if representation == "pga_feature"
            else ("true PGA sandwich-action adapter" if representation in {"pga_sandwich", "motordock_pga"} else representation)
        ),
    )
