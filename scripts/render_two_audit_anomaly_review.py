"""Server-only static review of phase-3 evidence; never changes candidate decisions."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from instance_repair.two_audit_cache import ObservationCache, write_json
from instance_repair.two_audit_sources import file_record, verify_record

PALETTE = [(30, 150, 255), (255, 135, 25), (35, 215, 145), (210, 75, 220),
           (245, 210, 40), (235, 80, 110), (70, 205, 225), (170, 195, 50)]


def choose_cards(pairs):
    by_id = defaultdict(list)
    for pair in pairs:
        if pair["status"] == "POTENTIAL_OVERMERGE":
            by_id[pair["target_instance_id"]].append(pair)
    cards = []
    for iid, candidates in sorted(by_id.items()):
        ranked = sorted(candidates, key=lambda p: (-p["independent_joint_view_groups"],
                          -p["joint_frame_count"], p["pair_key"]))
        selected = [ranked[0]]
        outside = [p for p in ranked if p["roi_comparison"]["outside_all_selected_roi_context"]]
        if outside and outside[0]["pair_key"] != selected[0]["pair_key"]:
            selected.append(outside[0])
        for pair in selected:
            cards.append(dict(pair=pair, target_id=iid, instance_pair_count=len(candidates),
                              instance_outside_pair_count=len(outside)))
    return cards


def render(audit_root, output):
    audit_root, output = Path(audit_root).resolve(strict=True), Path(output).absolute()
    if output.exists():
        raise FileExistsError("Review output already exists")
    if output.resolve() == audit_root or not output.resolve().is_relative_to(audit_root):
        raise ValueError("Review must be in a new child directory of this audit output")
    m = json.loads((audit_root/"manifest.json").read_text())
    if m.get("status") != "COMPLETE":
        raise ValueError("Review requires a completed audit")
    inputs = [file_record(audit_root/"manifest.json")]
    for name, digest in m["artifact_sha256"].items():
        record = file_record(audit_root/name)
        if record["sha256"] != digest:
            raise ValueError("Audit artifact changed")
        inputs.append(record)
    cache = ObservationCache(m["cache_root"])
    config_path = Path(cache.manifest["mapping_root"])/"config.json"
    cfg = json.loads(config_path.read_text())
    inputs.append(file_record(config_path))
    raw = cfg["source"]
    metadata_path = Path(raw["mask_root"])/raw["mask_metadata"]
    metadata_record = file_record(metadata_path)
    metadata = {}
    # Read only the source prefix represented by this cache, never future RGB.
    with metadata_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame"] in cache.frames:
                metadata[row["frame"]] = row
            if len(metadata) == len(cache.frames):
                break
    pairs = json.loads((audit_root/"candidate_pairs.json").read_text())
    cards = choose_cards(pairs)
    # A graph coloring keeps a track's color fixed and separates displayed neighbors.
    neighbors = defaultdict(set)
    for card in cards:
        a, b = card["pair"]["track_ids"]
        neighbors[a].add(b)
        neighbors[b].add(a)
    colors = {}
    for tid in sorted(neighbors, key=lambda key: (-len(neighbors[key]), key)):
        forbidden = {colors[n] for n in neighbors[tid] if n in colors}
        color = next((i for i in range(len(PALETTE)) if i not in forbidden), None)
        if color is None:
            # Deterministic extra hue for a dense display graph.
            from colorsys import hsv_to_rgb
            color = len(PALETTE)
            PALETTE.append(tuple(int(255*x) for x in hsv_to_rgb((color*.61803398875)%1, .78, 1)))
        colors[tid] = color
    output.mkdir(parents=True, exist_ok=False)
    (output/"images").mkdir()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    pages, records, raw_records = [], [], {}
    start = time.perf_counter()
    for card_index, card in enumerate(cards):
        pair = card["pair"]
        events = {e["frame_id"]:e for e in pair["joint_observations"]}
        figures = []
        for group in pair["joint_view_groups"][:2]:
            fid = group["representative_frame_id"]
            event = events[fid]
            mask_record = cache.frames[fid]["sources"]["mask"]
            verify_record(mask_record)
            rgb_path = Path(raw["scene_root"])/raw["rgb_pattern"].format(frame=fid)
            rgb_record = file_record(rgb_path)
            if (metadata[fid]["input_sha256"] != rgb_record["sha256"]
                    or metadata[fid]["mask_sha256"] != mask_record["sha256"]):
                raise ValueError("RGB/Mask pair differs from original segmentation input")
            with Image.open(rgb_path) as source:
                rgb = np.asarray(source.convert("RGB")).copy()
            with Image.open(mask_record["path"]) as source:
                labels = np.asarray(source).copy()
            if rgb.shape[:2] != labels.shape or list(labels.shape) != metadata[fid]["shape"]:
                raise ValueError("RGB/Mask shapes differ")
            selected = [labels == mid for mid in event["mask_local_ids"]]
            if not all(mask.any() for mask in selected) or np.any(selected[0]&selected[1]):
                raise ValueError("Review masks absent or overlapping")
            overlay = rgb.astype(np.float32)
            for tid, mask in zip(pair["track_ids"], selected):
                color = np.array(PALETTE[colors[tid]], np.float32)
                overlay[mask] = .53*overlay[mask] + .47*color
                inner = mask.copy()
                inner[1:] &= mask[:-1]
                inner[:-1] &= mask[1:]
                inner[:, 1:] &= mask[:, :-1]
                inner[:, :-1] &= mask[:, 1:]
                overlay[mask & ~inner] = color
            overlay = Image.fromarray(overlay.astype(np.uint8))
            yy, xx = np.where(selected[0] | selected[1])
            margin = 60
            box = (max(0, int(xx.min())-margin), max(0, int(yy.min())-margin),
                   min(rgb.shape[1], int(xx.max())+margin+1), min(rgb.shape[0], int(yy.max())+margin+1))
            canvas = Image.new("RGB", (1600, 710), (245, 247, 250))
            full = ImageOps.pad(overlay, (800, 600), color=(20, 22, 26))
            detail = ImageOps.pad(overlay.crop(box), (800, 600), color=(20, 22, 26))
            canvas.paste(full, (0, 110))
            canvas.paste(detail, (800, 110))
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 8), f"{m['scene']} | old ID {pair['target_instance_id']} | raw frame {fid} | view group {group['view_group_id']}", fill="black", font=font)
            draw.text((15, 35), "Original RGB + raw masks (left); enlarged mask region (right). Candidate is UNREVIEWED.", fill="black", font=small)
            for i, tid in enumerate(pair["track_ids"]):
                text = f"{tid} / mask {event['mask_local_ids'][i]}: old-ID support {event['target_support_points'][i]}/{event['mask_support_points'][i]} = {event['target_support_fractions'][i]:.2%}"
                draw.text((15+i*800, 69), text, fill=PALETTE[colors[tid]], font=small)
            name = f"images/{card_index:03d}_{pair['pair_key'][:12]}_f{fid:06d}.jpg"
            canvas.save(output/name, quality=90)
            with Image.open(output/name) as check:
                check.verify()
            figures.append(name)
            records.append(dict(image=name, image_sha256=file_record(output/name)["sha256"],
                pair_key=pair["pair_key"], frame_id=fid, joint_view_group_id=group["view_group_id"],
                track_ids=pair["track_ids"], mask_local_ids=event["mask_local_ids"], crop_xyxy=box,
                rgb=rgb_record, mask=mask_record))
            raw_records[rgb_record["path"]] = rgb_record
            raw_records[mask_record["path"]] = mask_record
        r = pair["roi_comparison"]
        full = r["full_mask_footprint"]
        body = f"<article><h2>Old ID {pair['target_instance_id']} · {html.escape(' / '.join(pair['track_ids']))}</h2>"
        body += f"<p>This old ID has {card['instance_pair_count']} candidate pairs; {card['instance_outside_pair_count']} outside all selected ROI context. This pair: {pair['joint_frame_count']} joint frames, {pair['independent_joint_view_groups']} view groups.</p>"
        body += f"<p>ROI footprint: {full['relation']}; core {full['core_fraction']:.2%}, context {full['context_fraction']:.2%}. Minimum joint old-ID support fraction: {pair['minimum_joint_target_fraction']:.2%}.</p>"
        body += "".join(f"<a href='{name}'><img src='{name}' loading='lazy'></a>" for name in figures)
        body += f"<p class='key'>pair_key: {pair['pair_key']}</p></article>"
        pages.append(body)
    for record in inputs+list(raw_records.values())+[metadata_record]:
        verify_record(record)
    page = "<!doctype html><html><head><meta charset='utf-8'><title>Two-audit phase 3 review</title><style>body{font:16px system-ui;margin:32px;background:#f1f4f8;color:#152334}article{background:white;padding:20px;margin:22px 0;border-radius:12px}img{max-width:100%;display:block;margin:12px 0}h1,h2{line-height:1.3}.key{font:12px monospace;overflow-wrap:anywhere}</style></head><body>"
    page += f"<h1>{html.escape(m['scene'])}: potential overmerge review</h1><p>Original raw-mask evidence on the fixed pre-repair map. No confirmed error labels, no coverage-ratio gate, no map changes or repair. ROI geometric overlap does not establish that an error was detected or a seed accepted. Representative cards only; the full candidate list remains in candidate_pairs.json and pair_summary.csv.</p>"
    page += "".join(pages)+"</body></html>"
    (output/"index.html").write_text(page, encoding="utf-8")
    report = dict(status="PASS", scene=m["scene"], audit_run_id=m["run_id"],
        cards=len(cards), images=len(records), figures=records,
        representative_selection="most_joint_views_per_instance_plus_first_outside_roi_pair_if_different",
        view_selection="first_two_fixed_independent_joint_view_representatives",
        source_rgb_verified_against_original_mask_metadata=True,
        image_format_verified=True, human_visual_review_performed=False,
        decisions_changed=False, renderer=file_record(Path(__file__)),
        inputs=inputs, metadata_source=metadata_record,
        index=file_record(output/"index.html"), seconds=time.perf_counter()-start)
    write_json(output/"render_validation.json", report)
    return {k:report[k] for k in ("status", "scene", "cards", "images", "seconds")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.audit, args.output)))
