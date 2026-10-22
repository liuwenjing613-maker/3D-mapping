#!/usr/bin/env python3
"""Render actual CropFormer, OVI geometry and MaskFusion masks for inspection."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HOME = Path('/home/chenkejun/CVPR/revisable_instance_map')
DATA = Path('/data/chenkejun/CVPR/revisable_instance_map')


def colors(ids, shift=0):
    result = {}
    for idx in ids:
        idx = int(idx)
        if idx <= 0:
            continue
        hue = (idx * 47 + shift) % 180
        hsv = np.uint8([[[hue, 180, 240]]])
        result[idx] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return result


def overlay(rgb, labels, shift=0, alpha=.58, outlines=True):
    out = (rgb.astype(np.float32) * .50).astype(np.uint8)
    pal = colors(np.unique(labels), shift)
    for idx, color in pal.items():
        region = labels == idx
        out[region] = np.clip((1 - alpha) * rgb[region] + alpha * color, 0, 255).astype(np.uint8)
    if outlines:
        edge = np.zeros(labels.shape, np.bool_)
        edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edge[1:, :] |= labels[1:, :] != labels[:-1, :]
        out[edge & (labels > 0)] = (245, 245, 245)
    return out


def label(panel, title, subtitle=None):
    h, w = panel.shape[:2]
    canvas = np.full((h + 73, w, 3), (22, 25, 30), np.uint8)
    cv2.putText(canvas, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, .78, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .49, (180, 208, 218), 1, cv2.LINE_AA)
    canvas[73:] = panel
    return canvas


def outline_parent(panel, raw, parent_id, thickness=3):
    mask = np.uint8(raw == parent_id)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 255, 255), thickness)
    return panel


def load(frame_id):
    raw_cfg = json.loads((HOME / 'configs/replica_room0_stride5.json').read_text())
    scene = Path(raw_cfg['source']['scene_root'])
    raw_root = Path(raw_cfg['source']['mask_root'])
    rgb = cv2.imread(str(scene / raw_cfg['source']['rgb_pattern'].format(frame=frame_id)))
    raw = cv2.imread(str(raw_root / raw_cfg['source']['mask_pattern'].format(frame=frame_id)), cv2.IMREAD_UNCHANGED)
    geom = cv2.imread(str(DATA / 'ovimap_depth_room0_stride5' / f'{frame_id:05d}_mask.png'), cv2.IMREAD_UNCHANGED)
    fused = cv2.imread(str(DATA / 'ovimap_refined_room0_stride5' / f'frame{frame_id:06d}.png'), cv2.IMREAD_UNCHANGED)
    if any(x is None for x in (rgb, raw, geom, fused)):
        raise ValueError('An input image is missing')
    if rgb.shape[:2] != raw.shape or raw.shape != geom.shape or geom.shape != fused.shape:
        raise ValueError('Input dimensions differ')
    return rgb, raw, geom, fused


def make(frame_id, parent_id, outdir):
    rgb, raw, geom, fused = load(frame_id)
    parent = raw == parent_id
    if not np.any(parent):
        raise ValueError('Selected parent is absent')
    child_ids, child_counts = np.unique(fused[parent], return_counts=True)
    child_positive = [(int(i), int(c)) for i, c in zip(child_ids, child_counts) if i > 0]
    residual = int(np.count_nonzero(parent & (fused == 0)))
    crossing = int(np.count_nonzero((fused > 0) & (raw > 0) & (raw != parent_id)))
    panels = [rgb.copy(), overlay(rgb, raw, 5), overlay(rgb, geom, 65), overlay(rgb, fused, 120)]
    panels = [outline_parent(p, raw, parent_id) for p in panels]
    subtitles = [f'Frame {frame_id} / yellow = CropFormer id {parent_id}',
                 f'{len(np.unique(raw)) - 1} raw proposals',
                 f'{len(np.unique(geom)) - 1} geometric regions',
                 f'{len(np.unique(fused)) - 1} fused fragments']
    titles = ['1  RGB input', '2  CropFormer mask', '3  Depth geometry mask', '4  OVI MaskFusion output']
    full = np.concatenate([label(cv2.resize(p, (600, 340)), title, subtitle)
                           for p, title, subtitle in zip(panels, titles, subtitles)], axis=1)
    outdir.mkdir(parents=True, exist_ok=True)
    overview_path = outdir / f'frame{frame_id:06d}_overview.jpg'
    cv2.imwrite(str(overview_path), full, [cv2.IMWRITE_JPEG_QUALITY, 88])

    yy, xx = np.where(parent)
    x0, x1 = max(0, int(xx.min()) - 35), min(raw.shape[1], int(xx.max()) + 36)
    y0, y1 = max(0, int(yy.min()) - 35), min(raw.shape[0], int(yy.max()) + 36)
    crop = (slice(y0, y1), slice(x0, x1))
    crgb, craw, cgeom, cfused = [x[crop] for x in (rgb, raw, geom, fused)]
    focus = [crgb.copy(), overlay(crgb, craw, 5), overlay(crgb, cgeom, 65), overlay(crgb, cfused, 120)]
    focus = [outline_parent(p, craw, parent_id, thickness=2) for p in focus]
    target_h = 560
    target_w = max(400, round(crgb.shape[1] * target_h / crgb.shape[0]))
    focus = [cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_NEAREST) for p in focus]
    notes = ['Same crop in the RGB frame', f'One parent proposal: raw id {parent_id}',
             'One depth region can cut across parents',
             f'{len(child_positive)} positive fragments in parent; {residual} residual px']
    detail = np.concatenate([label(p, title, note) for p, title, note in zip(focus, titles, notes)], axis=1)
    focus_path = outdir / f'frame{frame_id:06d}_focus_parent{parent_id}.jpg'
    cv2.imwrite(str(focus_path), detail, [cv2.IMWRITE_JPEG_QUALITY, 91])

    regrouped_path = DATA / 'ovimap_refined_regrouped_by_source_room0_stride5' / f'frame{frame_id:06d}.png'
    regrouped = cv2.imread(str(regrouped_path), cv2.IMREAD_UNCHANGED)
    if regrouped is None or regrouped.shape != raw.shape:
        raise ValueError('Source-regrouped mask is missing or has wrong shape')
    group_panel = overlay(rgb, regrouped, 5)
    group_panel[parent & (regrouped == 0)] = (25, 165, 235)
    group_panel = outline_parent(group_panel, raw, parent_id)[crop]
    group_panel = cv2.resize(group_panel, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    grouped = np.concatenate([label(p, title, note) for p, title, note in zip(
        [focus[1], focus[3], group_panel],
        ['Raw CropFormer', 'OVI MaskFusion fragments', 'Our source-regrouped evidence'],
        ['One parent proposal', 'Many fragment IDs; majority-source labels',
         'Same raw-ID colors; orange = residual, still stored'])], axis=1)
    grouped_path = outdir / f'frame{frame_id:06d}_regrouped_parent{parent_id}.jpg'
    cv2.imwrite(str(grouped_path), grouped, [cv2.IMWRITE_JPEG_QUALITY, 91])

    # Explain where final fragment IDs disagree with the raw source ID.
    source_panel = np.full_like(rgb, (35, 38, 43))
    source_panel[parent & (fused > 0)] = (190, 210, 45)  # source retained: teal
    source_panel[parent & (fused == 0)] = (25, 165, 235)  # raw residual: orange
    source_panel[(raw != parent_id) & (fused > 0)] = (75, 75, 75)
    source_panel = outline_parent(source_panel, raw, parent_id)
    source_panel = source_panel[crop]
    source_panel = cv2.resize(source_panel, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    source_path = outdir / f'frame{frame_id:06d}_source_evidence_parent{parent_id}.jpg'
    cv2.imwrite(str(source_path), label(source_panel, 'Source evidence', 'Teal = retained; orange = residual raw pixels'), [cv2.IMWRITE_JPEG_QUALITY, 91])
    report = {'frame_id': frame_id, 'parent_raw_id': parent_id, 'parent_pixels': int(np.count_nonzero(parent)),
              'parent_bbox_xyxy': [x0, y0, x1, y1], 'child_fragments': child_positive,
              'residual_parent_pixels': residual, 'all_other_parent_positive_pixels': crossing,
              'files': [str(overview_path), str(focus_path), str(source_path), str(grouped_path)],
              'input_sha256': {
                  'cropformer': hashlib.sha256((Path(json.loads((HOME / 'configs/replica_room0_stride5.json').read_text())['source']['mask_root']) / f'frame{frame_id:06d}.png').read_bytes()).hexdigest(),
                  'depth_geometry': hashlib.sha256((DATA / 'ovimap_depth_room0_stride5' / f'{frame_id:05d}_mask.png').read_bytes()).hexdigest(),
                  'maskfusion': hashlib.sha256((DATA / 'ovimap_refined_room0_stride5' / f'frame{frame_id:06d}.png').read_bytes()).hexdigest(),
                  'source_regrouped': hashlib.sha256(regrouped_path.read_bytes()).hexdigest(),
              }}
    (outdir / f'frame{frame_id:06d}_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def make_cross_source(frame_id, segment_id, outdir):
    rgb, raw, geom, fused = load(frame_id)
    mask = np.uint8(fused == segment_id)
    if not np.any(mask):
        raise ValueError('Selected fused segment is absent')
    segment_records = [json.loads(line) for line in
        (DATA / 'ovimap_refined_room0_stride5/segments.jsonl').open()
        if json.loads(line)['frame_id'] == frame_id and json.loads(line)['local_id'] == segment_id]
    if len(segment_records) != 1:
        raise ValueError('Segment metadata is missing or duplicated')
    parent_id = int(segment_records[0]['cropformer_id'])
    same = (mask > 0) & (raw == parent_id)
    other = (mask > 0) & (raw > 0) & (raw != parent_id)
    background = (mask > 0) & (raw == 0)
    heat = (rgb.astype(np.float32) * .22).astype(np.uint8)
    heat[same] = (205, 200, 42)        # teal in BGR
    heat[other] = (35, 55, 235)        # red
    heat[background] = (35, 175, 245) # orange
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_rgb = rgb.copy()
    cv2.drawContours(selected_rgb, contours, -1, (255, 0, 255), 3)
    raw_panel = overlay(rgb, raw, 5)
    cv2.drawContours(raw_panel, contours, -1, (255, 0, 255), 3)
    cv2.drawContours(heat, contours, -1, (255, 0, 255), 3)
    panels = [selected_rgb, raw_panel, heat]
    titles = ['RGB / one fused region', 'Raw CropFormer sources', 'Actual pixel origins']
    notes = [f'Frame {frame_id}, fused fragment {segment_id} in magenta',
             f'MaskFusion assigns parent id {parent_id} by majority',
             f'Teal: {np.count_nonzero(same)} same  Red: {np.count_nonzero(other)} other  Orange: {np.count_nonzero(background)} background']
    result = np.concatenate([label(cv2.resize(panel, (700, 397)), title, note)
                             for panel, title, note in zip(panels, titles, notes)], axis=1)
    path = outdir / f'frame{frame_id:06d}_cross_source_segment{segment_id}.jpg'
    cv2.imwrite(str(path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps({'cross_source_visual': str(path), 'assigned_parent': parent_id,
                      'same_pixels': int(np.count_nonzero(same)),
                      'other_source_pixels': int(np.count_nonzero(other)),
                      'background_pixels': int(np.count_nonzero(background))}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-id', type=int, required=True)
    parser.add_argument('--parent-id', type=int, required=True)
    parser.add_argument('--segment-id', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    make(args.frame_id, args.parent_id, args.output_dir)
    if args.segment_id is not None:
        make_cross_source(args.frame_id, args.segment_id, args.output_dir)
