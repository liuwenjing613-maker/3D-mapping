"""Complementary class-agnostic metrics on the unchanged V3 reference masks."""
from pathlib import Path
import os
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

ROOT = Path(os.environ.get('CVPR_EVAL_AUDIT_OUT', '/data/chenkejun/CVPR/results/evaluation_audit_20261007'))
OUT = Path(os.environ.get('CVPR_EXTENDED_METRICS_OUT', str(ROOT / 'extended_metrics')))
SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
THRESHOLDS = [f'{x / 100:.2f}' for x in range(50, 100, 5)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, location):
    assert np.isclose(actual, expected, atol=1e-12, rtol=1e-12), f'{location}: {actual} != {expected}'


def main():
    audit = read(ROOT / 'verification.json')
    assert audit['status'] == 'PASS' and audit['total_scene_evaluations'] == 40
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'date': '2026-10-07', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'DEBUG_ONLY / NON_OFFICIAL', 'protocol': 'Replica-CA-v3',
              'effective_protocol_sha256': audit['effective_protocol_sha256'],
              'configuration_and_reference_unchanged': True,
              'units': 'ratios [0,1], counts are integers',
              'aggregation': 'instance/vertex pooled; equal-weight scene macro reported separately',
              'definitions': {
                  'CA_mCov': 'For each eligible GT instance j, max_i IoU(P_i,G_j), then mean across all GT including missed instances as zero. All classes collapsed.',
                  'CA_mWCov': 'Same best-GT IoU, weighted by GT reference vertex count. Vertex weighted, not physical triangle-area weighted.',
                  'CA_mAR_all': 'Mean of TP_t / total_GT for IoU thresholds 0.50:0.05:0.95, strict > and V3 one-to-one matching, all predictions, no maxDet cap. Not COCO AR@100.',
                  'R25_R50_R75': 'TP/(TP+FN) at strict IoU>0.25/0.50/0.75 under the same V3 one-to-one matcher.',
                  'P25_P50_P75': 'TP/(TP+FP), excluding V3 void-majority unmatched predictions at each corresponding threshold.',
                  'SQ50': 'Mean IoU of correctly matched pairs at IoU>0.50; conditioned on true positives.',
                  'GT_surface_coverage': 'Assigned eligible GT vertices / all eligible GT vertices, regardless of instance identity.',
                  'Completeness': 'Per GT maximum intersection / GT size over any predicted instance; average across all GT including misses as zero.',
                  'Purity': 'Per eligible nonempty prediction maximum intersection / prediction size; mean across predictions, exclude void-majority predictions as in existing V3 diagnostics.',
                  'merge_split_duplicate': 'Existing V3 pairwise 2cm support diagnostics, with >=10 vertices and >=5% of GT; unchanged.'
              },
              'sources': [
                  {'topic': 'mCov/mWCov in 3D instance segmentation', 'title': 'Unified 3D Segmenter as Prototypical Classifiers, NeurIPS 2023', 'url': 'https://proceedings.neurips.cc/paper_files/paper/2023/file/916cb4e1aeafaa0757953c9bacd17337-Paper-Conference.pdf'},
                  {'topic': 'PQ=SQ*RQ and RQ=F1', 'title': 'Panoptic Segmentation, CVPR 2019', 'url': 'https://arxiv.org/html/1801.00868'},
                  {'topic': 'Average recall over IoU thresholds and maxDet conventions', 'title': 'COCO official evaluator', 'url': 'https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py'},
                  {'topic': 'VI split/merge diagnostics, optional future extension', 'title': 'scikit-image metrics documentation', 'url': 'https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.variation_of_information'},
                  {'topic': 'Temporal association metrics, requires GT/predicted track histories', 'title': 'TrackEval official HOTA implementation', 'url': 'https://github.com/JonathonLuiten/TrackEval'},
              ], 'methods': [], 'verification': []}
    for method in audit['methods']:
        key = method['method']
        summary = read(method['recomputed_summary'])
        expected_rows = {x['scene_id']: x for x in summary['per_scene']}
        scene_rows, best_iou_parts, gt_size_parts, purity_parts, recall_parts = [], [], [], [], []
        total_covered = 0
        matrix_evidence = []
        for evidence in method['evidence']:
            scene = evidence['scene']
            path = Path(evidence['evaluation_dir']) / 'overlap_matrix.npz'
            row = expected_rows[scene]
            with np.load(path, allow_pickle=False) as data:
                iou = data['iou']
                intersection = data['intersection']
                pred_size, gt_size = data['pred_size'], data['gt_size']
                assert len(gt_size) == row['diagnostics']['gt_instance_count']
                assert np.all(gt_size >= 100)
                union = pred_size[:, None] + gt_size[None, :] - intersection
                rebuilt_iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
                assert np.array_equal(iou, rebuilt_iou)
                # Main V3 masks partition the reference vertices, so columns cannot double count coverage.
                assert np.all(intersection.sum(axis=0) <= gt_size)
                best_iou = iou.max(axis=0) if len(iou) else np.zeros(len(gt_size))
                recall = data['recall'].max(axis=0) if len(iou) else np.zeros(len(gt_size))
                eligible = (pred_size > 0) & (data['pred_void_fraction'] <= 0.5)
                purity = data['precision'].max(axis=1)[eligible]
                covered = int(intersection.sum())
                stats = {
                    'scene': scene, 'GT_count': len(gt_size), 'GT_vertices': int(gt_size.sum()),
                    'CA_mCov': float(best_iou.mean()),
                    'CA_mWCov': float(np.dot(best_iou, gt_size) / gt_size.sum()),
                    'CA_mAR_all': float(np.mean([row['AP_by_threshold'][t]['TP'] / len(gt_size) for t in THRESHOLDS])),
                    'SQ50': row['CA_PQ']['SQ'],
                    'GT_surface_coverage': covered / int(gt_size.sum()),
                    'Completeness': float(recall.mean()),
                    'Purity': float(purity.mean()) if len(purity) else None,
                    'covered_GT_vertices': covered,
                    'purity_prediction_count': int(eligible.sum()),
                    'merge_prediction_rate': row['structure']['merge_prediction_rate'],
                    'split_gt_rate': row['structure']['split_gt_rate'],
                    'duplicate_prediction_rate': row['structure']['duplicate_prediction_rate'],
                }
                for suffix, threshold in [('25', '0.25'), ('50', '0.50'), ('75', '0.75')]:
                    counts = row['AP_by_threshold'][threshold]
                    tp, fp, fn = (counts[k] for k in ['TP', 'FP', 'FN'])
                    stats['R' + suffix] = tp / (tp + fn)
                    stats['P' + suffix] = tp / (tp + fp) if tp + fp else 0.0
                    stats['F1_' + suffix] = 2 * tp / (2 * tp + fp + fn)
                close(stats['CA_mCov'], row['diagnostics']['mean_best_gt_iou'], key + '/' + scene + '/mCov')
                close(stats['GT_surface_coverage'], row['diagnostics']['gt_surface_coverage'], key + '/' + scene + '/coverage')
                close(stats['Completeness'], row['diagnostics']['macro_best_gt_recall'], key + '/' + scene + '/completeness')
                close(stats['Purity'], row['diagnostics']['macro_prediction_purity'], key + '/' + scene + '/purity')
                close(stats['R50'], row['CA_PRF1_0_5']['R'], key + '/' + scene + '/R50')
                close(stats['P50'], row['CA_PRF1_0_5']['P'], key + '/' + scene + '/P50')
                close(stats['F1_50'], row['CA_PRF1_0_5']['F1'], key + '/' + scene + '/F1')
                close(row['CA_PQ']['RQ'], row['CA_PRF1_0_5']['F1'], key + '/' + scene + '/RQ=F1')
                best_iou_parts.append(best_iou.copy())
                gt_size_parts.append(gt_size.copy())
                purity_parts.append(purity.copy())
                recall_parts.append(recall.copy())
                total_covered += covered
                scene_rows.append(stats)
                matrix_evidence.append({'scene': scene, 'source': str(path), 'sha256': sha(path)})
        assert [x['scene'] for x in scene_rows] == SCENES
        best_iou, gt_size, purity, recall = (np.concatenate(parts) for parts in [best_iou_parts, gt_size_parts, purity_parts, recall_parts])
        n = len(gt_size)
        pooled = {
            'GT_count': n, 'GT_vertices': int(gt_size.sum()),
            'CA_mCov': float(best_iou.mean()),
            'CA_mWCov': float(np.dot(best_iou, gt_size) / gt_size.sum()),
            'CA_mAR_all': float(np.mean([summary['AP_by_threshold'][t]['TP'] / n for t in THRESHOLDS])),
            'SQ50': summary['CA_PQ']['SQ'],
            'GT_surface_coverage': total_covered / int(gt_size.sum()),
            'Completeness': float(recall.mean()), 'Purity': float(purity.mean()),
            'merge_prediction_rate': summary['structure']['merge_prediction_rate'],
            'split_gt_rate': summary['structure']['split_gt_rate'],
            'duplicate_prediction_rate': summary['structure']['duplicate_prediction_rate'],
        }
        for suffix, threshold in [('25', '0.25'), ('50', '0.50'), ('75', '0.75')]:
            tp, fp, fn = (summary['AP_by_threshold'][threshold][k] for k in ['TP', 'FP', 'FN'])
            pooled['R' + suffix] = tp / (tp + fn)
            pooled['P' + suffix] = tp / (tp + fp) if tp + fp else 0.0
            pooled['F1_' + suffix] = 2 * tp / (2 * tp + fp + fn)
        macro = {name: float(np.mean([x[name] for x in scene_rows])) for name, value in pooled.items() if isinstance(value, float)}
        close(pooled['R50'], summary['CA_PRF1_0_5']['R'], key + '/pooled/R50')
        close(pooled['P50'], summary['CA_PRF1_0_5']['P'], key + '/pooled/P50')
        result['methods'].append({'method': key, 'scope': 'reference only; different input budget' if key.startswith('OpenVox') else 'identical 400-frame input budget',
                                  'source_summary_sha256': sha(method['recomputed_summary']),
                                  'pooled': pooled, 'macro_per_scene': macro, 'per_scene': scene_rows})
        result['verification'].append({'method': key, 'status': 'PASS', 'source_overlap_matrices': matrix_evidence,
                                       'existing_diagnostics_and_R50_P50_F1_recomputed_consistently': True})
        print(json.dumps({'method': key, 'status': 'PASS', 'pooled': pooled}), flush=True)
    result['verification_status'] = 'PASS'
    (OUT / 'extended_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('ALL_PASS', OUT, flush=True)


if __name__ == '__main__':
    main()
