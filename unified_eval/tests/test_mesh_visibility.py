import numpy as np
import pytest

from unified_eval.mesh_visibility import GTMeshVisibility
from unified_eval.observable_surface import ObservationFrame, observed_reference_vertices_from_frame


def test_mesh_occlusion_and_mixed_label_faces_do_not_observe_hidden_objects():
    pytest.importorskip('open3d')
    xyz = np.array([[-1., -1., 1.], [1., -1., 1.], [0., 1., 1.],
                    [-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    raw = np.array([2, 2, 2, 1, 1, 1], dtype=np.int64)
    renderer = GTMeshVisibility(xyz, triangles, raw)
    ref = np.array([[0., 0., 2.]])
    ids, depth = renderer.render_reference_pixels(ref, np.eye(3), np.eye(4), (1, 1))
    assert ids[0, 0] == 2 and np.isclose(depth[0, 0], 1)
    frame = ObservationFrame(0, depth.copy(), np.eye(3), np.eye(4), ids, depth)
    assert not observed_reference_vertices_from_frame(ref, np.array([1]), frame, .02).any()
    mixed = GTMeshVisibility(xyz, triangles, np.array([2, 3, 2, 1, 1, 1]))
    ids, depth = mixed.render_reference_pixels(ref, np.eye(3), np.eye(4), (1, 1))
    assert ids[0, 0] == -1 and np.isclose(depth[0, 0], 1)
    assert mixed.mixed_label_face_count == 1


def test_raycast_depth_is_camera_z_without_normalizing_ray_directions():
    pytest.importorskip('open3d')
    xyz = np.array([[-6., -6., 2.], [6., -6., 2.], [0., 6., 2.]], dtype=np.float32)
    renderer = GTMeshVisibility(xyz, np.array([[0, 1, 2]]), np.array([4, 4, 4]))
    ids, depth = renderer.render_reference_pixels(np.array([[2., 0., 2.]]), np.eye(3), np.eye(4), (1, 2))
    assert ids[0, 1] == 4
    assert np.isclose(depth[0, 1], 2)


def test_replica_quads_are_triangulated_and_mixed_parent_labels_remain_unknown():
    pytest.importorskip('open3d')
    xyz = np.array([[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]], dtype=np.float32)
    faces = np.array([[0, 1, 2, 3]])
    renderer = GTMeshVisibility(xyz, faces, np.array([2, 2, 2, 2]))
    ids, depth = renderer.render_reference_pixels(np.array([[0., 0., 1.]]), np.eye(3), np.eye(4), (1, 1))
    assert ids[0, 0] == 2 and np.isclose(depth[0, 0], 1)
    mixed = GTMeshVisibility(xyz, faces, np.array([2, 2, 2, 3]))
    assert mixed.face_ids.tolist() == [-1, -1]
