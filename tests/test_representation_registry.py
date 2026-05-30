from motordock.models.representation_registry import get_representation_spec


def test_all_representations_registered():
    for r in ["se3_log","quaternion_translation","dual_quaternion","matrix","centroid_bias","random_motor","shuffled_pairs","no_pair_context","pga_feature","pga_sandwich"]:
        s = get_representation_spec(r)
        assert s.name == r


def test_specs_have_required_fields():
    s = get_representation_spec("se3_log")
    assert s.pair_feature_dim > 0 and s.predicted_dim > 0


def test_no_pair_context_spec_disables_transform_loss():
    s = get_representation_spec("no_pair_context")
    assert s.uses_transform_loss is False
