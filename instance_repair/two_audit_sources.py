"""Read immutable original observations for offline and causal online audits."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8"))


def file_record(path):
    path = Path(path).absolute()
    return dict(path=str(path), resolved_path=str(path.resolve(strict=True)),
                sha256=sha256_file(path))


def verify_record(record):
    path = Path(record["path"])
    if str(path.resolve(strict=True)) != record["resolved_path"]:
        raise ValueError(f"Source symlink changed: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Source content changed: {path}")


def geometry_identity(xyz):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("Geometry must be a nonempty N x 3 array")
    if xyz.dtype.kind != "f" or not np.isfinite(xyz).all():
        raise ValueError("Geometry must contain finite floating-point coordinates")
    # Shape, dtype and exact ordered bytes all participate; a reordered surface
    # is a different index space even if its unordered point set is identical.
    record = dict(shape=list(xyz.shape), dtype=xyz.dtype.str,
                  ordered_xyz_sha256=sha256_bytes(np.ascontiguousarray(xyz).tobytes()))
    return dict(record, geometry_id=canonical_hash(record))


def frame_ids(config):
    selection = config["frame_selection"]
    start, stop, stride = (int(selection[k]) for k in ("start", "stop_exclusive", "stride"))
    if start < 0 or stride < 1 or stop <= start:
        raise ValueError("Invalid frame selection")
    return tuple(range(start, stop, stride))


def read_matrix_prefix(path, count, dim):
    """Consume only the arrived matrix prefix, never the rest of a trajectory."""
    path = Path(path)
    target = count * dim * dim
    parts, size, raw_hash, byte_count = [], 0, hashlib.sha256(), 0
    with path.open("rb") as stream:
        while size < target:
            line = stream.readline()
            if not line:
                raise ValueError(f"Matrix source shorter than prefix: {path}")
            raw_hash.update(line)
            byte_count += len(line)
            clean = line.split(b"#", 1)[0].strip()
            if not clean:
                continue
            tokens = clean.split()
            if len(tokens) not in (dim, dim * dim) or size + len(tokens) > target:
                raise ValueError(f"Unsupported matrix row layout: {path}")
            parts.append(np.asarray([float(x) for x in tokens], dtype=np.float64))
            size += len(tokens)
    data = np.concatenate(parts).reshape(count, dim, dim)
    record = dict(path=str(path.absolute()), resolved_path=str(path.resolve(strict=True)),
                  prefix_sha256=raw_hash.hexdigest(), prefix_bytes=byte_count,
                  matrix_count=count, dimension=dim)
    return data, record


def verify_prefix(record):
    path = Path(record["path"])
    if str(path.resolve(strict=True)) != record["resolved_path"]:
        raise ValueError(f"Matrix source symlink changed: {path}")
    remaining, h = record["prefix_bytes"], hashlib.sha256()
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Matrix prefix shortened: {path}")
            h.update(chunk)
            remaining -= len(chunk)
    if h.hexdigest() != record["prefix_sha256"]:
        raise ValueError(f"Matrix prefix changed: {path}")


@dataclass(frozen=True)
class AuditFrame:
    frame_id: int
    depth_m: np.ndarray
    mask_local: np.ndarray
    pose: np.ndarray
    intrinsic: np.ndarray
    sources: dict
    observations: tuple


class MappingSource:
    """Adapter to an already completed full-depth mapping snapshot.

    Only original PNG label images are observations. Neither association IDs,
    frame_support persistent IDs nor repaired pixel journals are read.
    """
    def __init__(self, mapping_root, mode, checkpoint):
        self.root = Path(mapping_root).resolve(strict=True)
        if mode not in ("offline", "online-v2"):
            raise ValueError("mode must be offline or online-v2")
        self.mode = mode
        self.locked_files = {}
        self.frame_sources = []

        def locked_json(relative_or_absolute):
            path = Path(relative_or_absolute)
            if not path.is_absolute():
                path = self.root / path
            record = file_record(path)
            self.locked_files[str(path)] = record
            value = json.loads(path.read_text(encoding="utf-8"))
            verify_record(record)
            return value

        self.config = locked_json("config.json")
        self.receipt = locked_json("mapping_complete.json")
        self.plan = locked_json("mapping_plan.json")
        self.original_config = locked_json(self.plan["original_config"])
        self.frame_ids = frame_ids(self.config)
        original_ids = frame_ids(self.original_config)
        self.frame_set = frozenset(self.frame_ids)
        self.cutoff = self.frame_ids[-1]
        self.checkpoint = int(checkpoint)
        self.scene = str(self.config["scene"])
        self.dataset = str(self.config.get("source_dataset", self.config["dataset"]))
        self.model = str(self.config["mask"].get("provenance", ""))
        if not self.model or int(self.config["mask"]["background_id"]) != 0:
            raise ValueError("Original mask provenance and zero background are required")
        if self.scene != str(self.original_config["scene"]):
            raise ValueError("Original and mapping scene differ")
        if self.receipt.get("status") != "PASS":
            raise ValueError("Mapping has not completed successfully")
        for key in ("GT_used", "future_inputs_used", "future_baseline_used",
                    "current_checkpoint_repair_applied"):
            if self.receipt.get(key) is not False:
                raise ValueError(f"Mapping receipt must certify {key}=false")
        if self.receipt.get("native_shared_geometry") is not True:
            raise ValueError("A certified full-depth geometry snapshot is required")
        if self.receipt.get("geometry_incremental") is not True:
            raise ValueError("Incremental geometry provenance is required")
        if tuple(self.receipt["frame_ids"]) != self.frame_ids:
            raise ValueError("Mapping receipt and configured frames differ")
        if self.checkpoint != len(self.frame_ids) or self.receipt["frame_count"] != self.checkpoint:
            raise ValueError("Checkpoint is the number of processed mapping frames")
        if self.receipt["last_frame_id"] != self.cutoff or self.plan["raw_cutoff"] != self.cutoff:
            raise ValueError("Raw frame cutoff differs from the mapping receipt")
        if self.plan["target_processed_frames"] != self.checkpoint:
            raise ValueError("Mapping plan checkpoint mismatch")
        if self.frame_ids != original_ids[:self.checkpoint]:
            raise ValueError("Mapping frames must be an exact original sequence prefix")
        if mode == "offline":
            if self.frame_ids != original_ids:
                raise ValueError("Offline audit requires the full experimental sequence")
            if self.plan["prior_processed_frames"] != 0 or self.receipt.get("repair_applied") is not False:
                raise ValueError("Offline audit starts from the unrepaired full-scene map")
        elif not 0 <= self.plan["prior_processed_frames"] < self.checkpoint:
            raise ValueError("Invalid online prior checkpoint")
        if self.config.get("rules", {}).get("use_future_frames") is not False:
            raise ValueError("The input contract must prohibit future frames")

        surface = self.root / "baseline/native/instance_surface.npz"
        self.surface_source = file_record(surface)
        self.locked_files[str(surface)] = self.surface_source
        with np.load(surface, allow_pickle=False) as archive:
            self.xyz = archive["xyz_m"]
        self.geometry = geometry_identity(self.xyz)
        self.xyz.setflags(write=False)

        camera = self.config["camera"]
        self.width, self.height = int(camera["width"]), int(camera["height"])
        self.depth_scale = float(camera["depth_png_units_per_meter"])
        if min(self.width, self.height) <= 0 or not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("Invalid image dimensions or depth units")
        source = self.config["source"]
        self.scene_root, self.mask_root = Path(source["scene_root"]), Path(source["mask_root"])
        self.poses, pose_record = read_matrix_prefix(
            self.scene_root / source["trajectory"], self.cutoff + 1, 4)
        self.prefix_sources = [pose_record]
        self.intrinsics = None
        if camera.get("per_frame_intrinsics"):
            self.intrinsics, intr_record = read_matrix_prefix(
                self.scene_root / camera["per_frame_intrinsics"], self.cutoff + 1, 3)
            self.prefix_sources.append(intr_record)
        self.static_intrinsic = np.array([[camera["fx"], 0, camera["cx"]],
                                          [0, camera["fy"], camera["cy"]],
                                          [0, 0, 1]], dtype=np.float64)

        ledger_path = self.root / "association/observations.jsonl"
        self.locked_files[str(ledger_path)] = file_record(ledger_path)
        self.by_frame = {fid: {} for fid in self.frame_ids}
        with ledger_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                fid, mid = int(row["frame_id"]), int(row["mask_local_id"])
                if fid not in self.frame_set or mid <= 0 or mid in self.by_frame[fid]:
                    raise ValueError("Raw observation ledger contains out-of-prefix/duplicate IDs")
                self.by_frame[fid][mid] = row
        verify_record(self.locked_files[str(ledger_path)])

        self.mask_metadata = {}
        metadata_path = self.mask_root / source["mask_metadata"]
        # The metadata may be shared with a longer run. Stop at this prefix.
        metadata_hash, metadata_bytes = hashlib.sha256(), 0
        with metadata_path.open("rb") as stream:
            previous = -1
            for line in stream:
                metadata_hash.update(line)
                metadata_bytes += len(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                fid = int(row["frame"])
                if fid <= previous:
                    raise ValueError("Original mask metadata must have unique ordered frames")
                previous = fid
                if fid > self.cutoff:
                    break
                if fid in self.frame_set:
                    self.mask_metadata[fid] = row
                if fid == self.cutoff:
                    break
        if set(self.mask_metadata) != self.frame_set:
            raise ValueError("Missing original mask metadata for the arrived prefix")
        self.metadata_source = dict(path=str(metadata_path.absolute()),
                                    resolved_path=str(metadata_path.resolve(strict=True)),
                                    prefix_sha256=metadata_hash.hexdigest(),
                                    prefix_bytes=metadata_bytes,
                                    selected_rows_sha256=canonical_hash(self.mask_metadata))
        self.prefix_sources.append(self.metadata_source)

    def load_frame(self, fid, max_depth_m):
        if fid not in self.frame_set:
            raise ValueError(f"Frame {fid} lies outside the arrived prefix")
        source = self.config["source"]
        original = self.original_config["source"]
        mask_path = self.mask_root / source["mask_pattern"].format(frame=fid)
        raw_path = Path(original["mask_root"]) / original["mask_pattern"].format(frame=fid)
        if mask_path.resolve(strict=True) != raw_path.resolve(strict=True):
            raise ValueError("Mapping mask does not resolve to the original mask source")
        depth_path = self.scene_root / source["depth_pattern"].format(frame=fid)
        raw_depth_path = Path(original["scene_root"]) / original["depth_pattern"].format(frame=fid)
        if depth_path.resolve(strict=True) != raw_depth_path.resolve(strict=True):
            raise ValueError("Mapping depth does not resolve to the original depth source")
        sources, arrays = {}, {}
        for kind, path in (("mask", mask_path), ("depth", depth_path)):
            payload = path.read_bytes()
            sources[kind] = dict(path=str(path.absolute()),
                                 resolved_path=str(path.resolve(strict=True)),
                                 sha256=sha256_bytes(payload))
            with Image.open(io.BytesIO(payload)) as image:
                arrays[kind] = np.array(image)
        mask, depth_raw = arrays["mask"], arrays["depth"]
        shape = (self.height, self.width)
        if mask.shape != shape or depth_raw.shape != shape:
            raise ValueError("Raw mask/depth shape differs from the camera contract")
        # Pillow can expose uint16 PNGs as int32. Enforce integer values/range.
        for kind, array in arrays.items():
            if array.dtype.kind not in "ui" or array.min() < 0 or array.max() > 65535:
                raise ValueError(f"{kind} must be a nonnegative 8/16-bit PNG array")
        mask = mask.astype(np.uint16, copy=False)
        depth = depth_raw.astype(np.float32) / self.depth_scale
        digest = sources["mask"]["sha256"]
        if self.mask_metadata[fid]["mask_sha256"] != digest:
            raise ValueError("Original mask hash differs from its inference metadata")
        count = np.bincount(mask.ravel(), minlength=1)
        ids = np.flatnonzero(count)[1:] if count[0] else np.flatnonzero(count)
        ids = ids[ids > 0]
        valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth_m)
        valid_count = np.bincount(mask[valid], minlength=len(count))
        ledger = self.by_frame[fid]
        if set(map(int, ids)) != set(ledger):
            raise ValueError("Decoded original masks differ from the raw observation ledger")
        edges = set(map(int, np.unique(np.concatenate(
            (mask[0], mask[-1], mask[:, 0], mask[:, -1])))))
        observations = []
        for mid in map(int, ids):
            old = ledger[mid]
            if old["source_mask_sha256"] != digest or old["pixel_count"] != int(count[mid]):
                raise ValueError("Original observation provenance/pixel count mismatch")
            # The projectable count in the original ledger uses the inherited 10 m limit.
            if max_depth_m == 10.0 and old["projectable_pixel_count"] != int(valid_count[mid]):
                raise ValueError("Original observation valid-depth count mismatch")
            identity = dict(dataset=self.dataset, scene=self.scene, frame_id=fid,
                            mask_local_id=mid, source_mask_sha256=digest, source_model=self.model)
            observations.append(dict(identity, observation_key=canonical_hash(identity),
                                     original_observation_id=old["observation_id"],
                                     pixel_count=int(count[mid]),
                                     valid_depth_pixels=int(valid_count[mid]),
                                     depth_valid_ratio=float(valid_count[mid] / count[mid]),
                                     bbox_xyxy_exclusive=old["bbox_xyxy_exclusive"],
                                     touches_image_border=mid in edges))
        pose = self.poses[fid].copy()
        intrinsic = (self.intrinsics[fid] if self.intrinsics is not None
                     else self.static_intrinsic).copy()
        if (not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-3)):
            raise ValueError(f"Invalid rigid camera pose at frame {fid}")
        if (not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1])
                or intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0):
            raise ValueError(f"Invalid supported pinhole intrinsics at frame {fid}")
        for array in (depth, mask, pose, intrinsic):
            array.setflags(write=False)
        self.frame_sources.append(sources)
        return AuditFrame(fid, depth, mask, pose, intrinsic, sources, tuple(observations))

    def verify_unchanged(self):
        for record in self.locked_files.values():
            verify_record(record)
        for record in self.prefix_sources:
            verify_prefix(record)
        for sources in self.frame_sources:
            for record in sources.values():
                verify_record(record)
