"""GT triangle visibility at the camera pixels used by the reference evaluator."""
from __future__ import annotations

import numpy as np

from .io import sha256_array
from .schema import EvaluationError


class GTMeshVisibility:
    """Cast one camera ray per queried pixel through the entire source GT mesh.

    Faces with inconsistent vertex IDs still occlude other faces, but their ID is
    unknown. Rays are independent of prediction geometry and instance labels.
    """

    def __init__(self, xyz: np.ndarray, faces: np.ndarray, raw_ids: np.ndarray):
        import open3d as o3d

        xyz, faces, raw_ids = np.asarray(xyz), np.asarray(faces), np.asarray(raw_ids)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise EvaluationError("GT mesh coordinates must be finite [N,3]")
        if faces.ndim != 2 or faces.shape[1] not in (3, 4) or not np.issubdtype(faces.dtype, np.integer):
            raise EvaluationError("GT mesh requires integer triangular or quadrilateral faces")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(xyz)):
            raise EvaluationError("GT triangle index outside reference vertices")
        if raw_ids.shape != (len(xyz),) or not np.issubdtype(raw_ids.dtype, np.integer):
            raise EvaluationError("GT mesh raw IDs must correspond to reference vertices")
        face_ids = raw_ids[faces]
        self.face_ids = np.where(np.all(face_ids == face_ids[:, :1], axis=1), face_ids[:, 0], -1)
        # Replica source PLYs contain quads. A deterministic 0-2 diagonal keeps
        # their vertex geometry; mixed-label source polygons stay unknown in BOTH children.
        triangles = faces
        if faces.shape[1] == 4:
            triangles = np.concatenate((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]))
            self.face_ids = np.tile(self.face_ids, 2)
        self.source_hashes = {"mesh_xyz_sha256": sha256_array(xyz),
            "source_faces_sha256": sha256_array(faces), "triangles_sha256": sha256_array(triangles),
            "raw_instance_sha256": sha256_array(raw_ids),
            "triangulation": "source_triangles_or_quad_0_2_diagonal"}
        self.mixed_label_face_count = int(np.count_nonzero(self.face_ids < 0))
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(xyz.astype(np.float32)),
                                          o3d.core.Tensor(triangles.astype(np.uint32)))
        self.scene.add_triangles(mesh)

    def render_reference_pixels(self, ref_xyz: np.ndarray, intrinsics: np.ndarray,
                                world_from_camera: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Sparse z-buffer queries have the same result as rendering every pixel.

        Camera ray directions have camera-z=1, so ray t_hit is metric camera-z
        depth, matching Replica depth PNGs; directions must not be normalized.
        """
        import open3d as o3d

        ref, k, pose = np.asarray(ref_xyz), np.asarray(intrinsics), np.asarray(world_from_camera)
        if ref.ndim != 2 or ref.shape[1] != 3 or not np.isfinite(ref).all():
            raise EvaluationError("Visibility reference must be finite [N,3]")
        if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
            raise EvaluationError("Visibility camera dimensions are invalid")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or not np.allclose(k[[0, 1], [1, 0]], 0):
            raise EvaluationError("Visibility renderer requires positive focal lengths and zero skew")
        if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4):
            raise EvaluationError("Visibility camera pose must be rigid")
        height, width = shape
        if height <= 0 or width <= 0:
            raise EvaluationError("Visibility image dimensions must be positive")
        camera = (ref - pose[:3, 3]) @ pose[:3, :3]
        positive = camera[camera[:, 2] > 0]
        uvw = positive @ k.T
        pixels = np.floor(uvw[:, :2] / uvw[:, 2:3] + .5).astype(np.int64)
        in_image = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        pixels = np.unique(pixels[in_image], axis=0)
        depth = np.full(shape, np.nan, dtype=np.float32)
        ids = np.full(shape, -1, dtype=np.int64)
        if not len(pixels):
            return ids, depth
        directions = np.column_stack(((pixels[:, 0] - k[0, 2]) / k[0, 0],
                                      (pixels[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(pixels)))) @ pose[:3, :3].T
        origins = np.broadcast_to(pose[:3, 3], directions.shape)
        rays = np.column_stack((origins, directions)).astype(np.float32)
        hits = self.scene.cast_rays(o3d.core.Tensor(rays))
        distance, faces = hits['t_hit'].numpy(), hits['primitive_ids'].numpy()
        valid = np.isfinite(distance) & (distance > 0) & (faces < len(self.face_ids))
        xy = pixels[valid]
        depth[xy[:, 1], xy[:, 0]] = distance[valid]
        ids[xy[:, 1], xy[:, 0]] = self.face_ids[faces[valid]]
        return ids, depth
