import numpy as np
import pytest

from unified_eval.ovimap import adapt_export
from unified_eval.schema import EvaluationError


def test_ovimap_independent_projection_retains_overlap_and_empty_instance(tmp_path):
    path = tmp_path / "ovi.npz"
    np.savez_compressed(path,
        xyz=np.array([[0., 0, 0], [0., 0, 0], [4., 0, 0]]),
        instance=np.array([0, 1, -1], dtype=np.int32),
        native_instance_ids=np.array([11, 12, 13], dtype=np.int32),
        classes=np.array([2, 3, -1], dtype=np.int32))
    mapped = adapt_export(path, np.array([[0., 0, 0], [1., 0, 0]]), 0.05,
        scene_id="room0", method_name="OVI-MAP", method_commit="test",
        protocol_version="Replica-CA-v2")
    pred = mapped.prediction
    assert [x.instance_uid for x in pred.instances] == ["ovi-id:11", "ovi-id:12", "ovi-id:13"]
    assert [x.vertex_indices.tolist() for x in pred.instances] == [[0], [0], []]
    assert pred.is_partition is False
    assert mapped.statistics["unassigned_source_vertices"] == 1
    assert mapped.statistics["overlapped_ref_vertices"] == 1


def test_ovimap_rejects_undeclared_native_id(tmp_path):
    path = tmp_path / "ovi.npz"
    np.savez_compressed(path, xyz=np.array([[0., 0, 0]]),
        instance=np.array([1], dtype=np.int32),
        native_instance_ids=np.array([11], dtype=np.int32))
    with pytest.raises(EvaluationError, match="outside"):
        adapt_export(path, np.array([[0., 0, 0]]), 0.05,
            scene_id="room0", method_name="OVI-MAP", method_commit="test",
            protocol_version="Replica-CA-v2")
