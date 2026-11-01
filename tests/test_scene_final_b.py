import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scene_completion.pipeline import (FRAME_IDS, finalize, manifest_from_commit, read,
                                       run_scene, validate_manifest, write_scene_complete)


class FinalSceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        angle = np.arange(6) * np.pi / 3
        xyz = np.r_[np.c_[.01*np.cos(angle), .01*np.sin(angle), np.zeros(6)], [[0., 0., 0.]]]
        rgb = np.zeros_like(xyz)
        labels = np.r_[np.full(6, 7), -1].astype(np.int32)
        evidence = dict(xyz_m=xyz, rgb=rgb, state=np.r_[np.full(6, 2), 0], top1_instance_id=labels)
        surface = dict(xyz_m=xyz, rgb=rgb, instance_id=labels, native_instance_ids=np.array([7]))
        self.inputs = {}
        for key, data in dict(surface=surface, native=surface, evidence=evidence,
                              normal_geometry=dict(xyz_m=xyz, normals=np.tile([0., 0., 1.], (7, 1)))).items():
            p = self.root / (key + '.npz')
            np.savez(p, **data)
            self.inputs[key] = p
        self.manifest = self.root / 'scene_complete.json'
        write_scene_complete(self.manifest, scene='room0', workflow='no_repair',
                             frame_ids=FRAME_IDS, inputs=self.inputs, final_processing_complete=True)
        self.out = self.root / 'B'

    def edit(self, **kwargs):
        m = read(self.manifest)
        m.update(kwargs)
        self.manifest.write_text(json.dumps(m))

    def test_final_only_and_immutable(self):
        before = {k: p.read_bytes() for k, p in self.inputs.items()}
        r = finalize(self.manifest, self.out)
        self.assertEqual(r['newly_assigned'], 1)
        with np.load(self.out / 'instance_surface.npz') as z:
            self.assertEqual(z['instance_id'].tolist(), [7]*7)
        self.assertEqual(before, {k: p.read_bytes() for k, p in self.inputs.items()})
        self.assertFalse(r['online_state_updated'])

    def test_prefixes_fail_before_output(self):
        for n in [20, 50, 100, 200, 300, 399]:
            self.edit(frame_ids=FRAME_IDS[:n])
            with self.assertRaisesRegex(ValueError, '400 scene frames'):
                finalize(self.manifest, self.out)
            self.assertFalse(self.out.exists())

    def test_missing_frame_with_same_length_fails(self):
        self.edit(frame_ids=FRAME_IDS[:-1] + [1990])
        with self.assertRaises(ValueError):
            finalize(self.manifest, self.out)

    def test_every_workflow_requires_finished_final_repair(self):
        for workflow in ['main', 'auto-v3-swap', 'online-v1', 'online-v2']:
            m = read(self.manifest)
            m.update(workflow=workflow, final_repair_complete=False)
            with self.assertRaisesRegex(ValueError, 'last repair'):
                validate_manifest(m)
            m['final_repair_complete'] = True
            validate_manifest(m)

    def test_checkpoint_and_pending_stages_rejected(self):
        for update in [dict(stage='checkpoint'), dict(pending_stages=['repair']),
                       dict(final_processing_complete=False), dict(completion_history=['B'])]:
            m = read(self.manifest)
            m.update(update)
            with self.assertRaises(ValueError):
                validate_manifest(m)

    def test_resume_does_not_recompute(self):
        first = finalize(self.manifest, self.out)
        with patch('scene_completion.pipeline.complete', side_effect=AssertionError('called twice')):
            self.assertEqual(first, finalize(self.manifest, self.out))

    def test_input_mutation_and_output_tamper_rejected(self):
        finalize(self.manifest, self.out)
        output = self.out / 'instance_surface.npz'
        output.write_bytes(output.read_bytes() + b'tamper')
        with self.assertRaisesRegex(ValueError, 'published artifact changed'):
            finalize(self.manifest, self.out)
        self.inputs['surface'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            finalize(self.manifest, self.out)

    def test_completed_B_cannot_be_reused_as_input(self):
        finalize(self.manifest, self.out)
        second = self.root / 'second.json'
        write_scene_complete(second, scene='room0', workflow='no_repair', frame_ids=FRAME_IDS,
                             inputs={**self.inputs, 'surface': self.out / 'instance_surface.npz'},
                             final_processing_complete=True)
        with self.assertRaisesRegex(ValueError, 'B output cannot'):
            finalize(second, self.root / 'B_again')

    def test_output_lock_blocks_competing_writer(self):
        self.out.mkdir()
        (self.out / '.finalizing').write_text('another process')
        with self.assertRaises(FileExistsError):
            finalize(self.manifest, self.out)
        self.assertEqual((self.out / '.finalizing').read_text(), 'another process')

    def test_runner_waits_for_final_producer_and_failure_never_publishes(self):
        fresh = self.root / 'producer_scene.json'
        code = 'from pathlib import Path; Path(%r).write_bytes(Path(%r).read_bytes())' % (str(fresh), str(self.manifest))
        result = run_scene(fresh, self.out, [sys.executable, '-c', code])
        self.assertEqual(result['status'], 'PASS')
        failed = self.root / 'failed.json'
        import subprocess
        with self.assertRaises(subprocess.CalledProcessError):
            run_scene(failed, self.root / 'failed_B', [sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertFalse((self.root / 'failed_B').exists())

    def test_commit_adapter_rejects_a_400_frame_mapping_before_last_repair(self):
        root = self.root / 'checkpoint'
        (root / 'mapping').mkdir(parents=True)
        (root / 'repair').mkdir()
        (root / 'mapping/mapping_complete.json').write_text(json.dumps(dict(status='PASS', frame_ids=FRAME_IDS)))
        (root / 'repair/commit_complete.json').write_text(json.dumps(dict(status='PASS', state='COMMITTED_BRANCH_ONLY', processed_frames=300, last_raw_frame=1495)))
        with self.assertRaisesRegex(ValueError, 'last scene repair'):
            manifest_from_commit(root, scene='room0', workflow='online-v2')

    def test_policy_and_frozen_settings_agree(self):
        import scene_completion.pipeline as pipeline
        from dataclasses import asdict
        policy = read(Path(pipeline.__file__).with_name('policy.json'))
        self.assertEqual(policy['settings'], asdict(pipeline.Settings()))
        self.assertEqual(set(policy['workflows']), pipeline.WORKFLOWS)
        self.assertFalse(policy['C_enabled'])

    def test_no_repair_command_line_publishes_after_producer(self):
        import subprocess
        receipt = self.root / 'mapping_complete.json'
        fresh = self.root / 'cli_scene.json'
        code = 'from pathlib import Path; Path(%r).write_text(%r)' % (
            str(receipt), json.dumps(dict(status='PASS', frame_ids=FRAME_IDS)))
        command = [sys.executable, '-B', '-m', 'scene_completion', 'no-repair',
                   '--scene', 'room0', '--mapping-receipt', str(receipt),
                   '--manifest', str(fresh), '--out', str(self.out)]
        for key, value in self.inputs.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        subprocess.run(command + ['--', sys.executable, '-c', code], check=True,
                       stdout=subprocess.DEVNULL)
        self.assertEqual(read(self.out / 'scene_finalization.json')['newly_assigned'], 1)


if __name__ == '__main__':
    unittest.main()
