"""Render the 20 predeclared real-data structure candidates for visual review."""
from pathlib import Path
import json
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_eval.io import load_gt, load_prediction
from unified_eval.repair_profile import write_json

OLD = Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009')
OUT = Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_audit_20261009')


def main():
    destination = OUT / 'structure_review'
    destination.mkdir(exist_ok=True)
    all_rows = []
    for scene in ('room0', 'room1'):
        gt = load_gt(OLD / 'mesh' / scene / 'gt.npz')
        pred = load_prediction(OUT / scene / 'P1-A1_holes_geodesic/canonical_prediction.npz')
        cases = json.loads((OUT / scene / 'structure_visual_review_cases.json').read_text())['cases']
        labels = np.full(gt.vertex_count, -1, dtype=int)
        for index, instance in enumerate(pred.instances):
            labels[instance.vertex_indices] = index
        for ordinal, case in enumerate(cases):
            selected = (gt.evaluation_region == 1) & np.isin(gt.instance_id, case['raw_gt_ids'])
            xyz = gt.xyz_ref[selected]
            if not len(xyz):
                raise RuntimeError('Review case has no trusted GT support')
            low, high = xyz.min(axis=0), xyz.max(axis=0)
            margin = np.maximum((high - low) * .1, .03)
            focus = np.all((gt.xyz_ref >= low - margin) & (gt.xyz_ref <= high + margin), axis=1)
            indices = np.flatnonzero(focus & (gt.evaluation_region == 1))
            if len(indices) > 20000:
                indices = indices[::int(np.ceil(len(indices) / 20000))]
            fig = plt.figure(figsize=(12, 5))
            for number, (name, values) in enumerate([('GT object IDs', gt.instance_id), ('Main fixed-surface prediction', labels)], 1):
                ax = fig.add_subplot(1, 2, number, projection='3d')
                ax.scatter(*gt.xyz_ref[indices].T, s=2, c='#bdbdbd', alpha=.2)
                for value in np.unique(values[selected]):
                    mask = values[indices] == value
                    color = 'black' if value < 0 else None
                    label = ('unassigned' if value < 0 else str(int(value)) if number == 1 else pred.instances[int(value)].instance_uid)
                    ax.scatter(*gt.xyz_ref[indices[mask]].T, s=4, c=color, label=label)
                ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
                ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
                ax.set_zlim(low[2] - margin[2], high[2] + margin[2])
                ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
                ax.view_init(elev=25, azim=-55)
                ax.set_title(name)
                ax.legend(fontsize=6, loc='upper left')
            title = f'{scene} case {ordinal + 1}: {case["candidate_type"]}; GT {case["raw_gt_ids"]}'
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            filename = f'{scene}_case_{ordinal + 1:02d}.png'
            fig.savefig(destination / filename, dpi=140)
            plt.close(fig)
            all_rows.append({**case, 'image': filename,
                'trusted_GT_vertex_counts': {str(raw): int(np.count_nonzero(selected & (gt.instance_id == raw))) for raw in case['raw_gt_ids']},
                'note': 'Fixed observed GT coordinates; actual partition labels. Candidate selection is not a human judgment.'})
    write_json(destination / 'cases.json', {'review_status': 'PENDING_VISUAL_REVIEW', 'case_count': len(all_rows), 'cases': all_rows})
    thumbnails = []
    for row in all_rows:
        image = Image.open(destination / row['image']).convert('RGB')
        image.thumbnail((840, 360))
        thumbnails.append(image)
    sheet = Image.new('RGB', (1680, 360 * int(np.ceil(len(thumbnails) / 2))), 'white')
    for index, image in enumerate(thumbnails):
        sheet.paste(image, (840 * (index % 2), 360 * (index // 2)))
    sheet.save(destination / '20_cases_contact_sheet.png')
    print(json.dumps({'case_count': len(all_rows), 'destination': str(destination)}))


if __name__ == '__main__':
    main()
