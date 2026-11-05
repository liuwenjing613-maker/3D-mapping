"""Prepare per-mask semantics for the complete legal ABC evidence pool."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from .seed_final_abc import read, save, require, load_context
from .seed_abc_evidence import Inputs
from .seed_abc_semantic_rules import summarize_mask


def prepare(context, classification, out):
    out = Path(out)
    inputs = Inputs(context, out)
    original = inputs.load(classification)
    if not inputs.seeds:
        save(out / "semantics.json", dict(status="PASS", records={}, source_hashes=inputs.locks))
        return out / "semantics.json"
    cfg = original["config"]
    require(bool(original["records"]), "Need the current semantic model/label metadata")
    labels = {str(c["id"]): c["name"] for c in original["records"][0]["categories"]}
    cached = {r["frame"]: r for r in original["frames"]}
    rows = {r["mask_id"]: r for r in original["records"]}
    result = {}
    model = processor = None
    for frame in sorted({s["frame"] for s in inputs.seeds.values()}):
        rgb = inputs.sr / inputs.cfg["source"]["rgb_pattern"].format(frame=frame)
        rgb_sha = inputs.lock(rgb)
        if frame in cached:
            info = cached[frame]
            cache = Path(classification).parent / info["cache"]
            require(rgb_sha == info["RGB_sha256"] and inputs.lock(cache) == info["cache_sha256"],
                    "Semantic cache source changed")
            with np.load(cache, allow_pickle=False) as z:
                prediction, confidence = z["labels"].copy(), z["confidence"].copy()
        else:
            import torch
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            if model is None:
                directory = Path(cfg["model_directory"])
                weights = list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
                require(any(inputs.lock(p) == cfg["weight_sha256"] for p in weights),
                        "Semantic model weights differ")
                for name in ("config.json", "preprocessor_config.json"):
                    inputs.lock(directory / name)
                torch.set_num_threads(2)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                processor = SegformerImageProcessor.from_pretrained(directory, local_files_only=True)
                model = SegformerForSemanticSegmentation.from_pretrained(
                    directory, local_files_only=True).cuda().float().eval()
            tensor = processor(images=Image.open(rgb).convert("RGB"), return_tensors="pt")
            with torch.inference_mode():
                logits = model(**{k: v.cuda() for k, v in tensor.items()}).logits.float()
                up = torch.nn.functional.interpolate(logits, size=inputs.shape,
                                                      mode="bilinear", align_corners=False)
                confidence, prediction = up.softmax(dim=1).max(dim=1)
                prediction = prediction[0].cpu().numpy().astype(np.uint8)
                confidence = confidence[0].cpu().numpy().astype(np.float32)
            cache = out / ("frame%06d.npz" % frame)
            np.savez_compressed(cache, labels=prediction, confidence=confidence)
            inputs.lock(cache)
        for mid, seed in inputs.seeds.items():
            if seed["frame"] != frame:
                continue
            row = summarize_mask(prediction, confidence, inputs.mask(mid), labels, cfg)
            if mid in rows:
                require(rows[mid]["mask_sha256"] == seed["mask_sha256"],
                        "Existing semantic record belongs to another mask")
                for key in ("high_excluded_pixels", "top_class", "exclude", "area"):
                    require(row[key] == rows[mid][key], "Original semantic result changed")
            result[mid] = {k: row[k] for k in ("area", "exclude", "top_class",
                "high_excluded_pixels", "high_excluded_fraction", "top_class_fraction", "decision")}
    inputs.check_sources()
    save(out / "semantics.json", dict(status="PASS", records=result, source_hashes=inputs.locks))
    return out / "semantics.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("context", "classification", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    print(prepare(load_context(a.context), a.classification, a.out))


if __name__ == "__main__":
    main()
