"""Safety and provenance checks for the frozen entry (no server data needed)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from instance_repair import cli


class FrozenEntryTests(unittest.TestCase):
    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / 'keep.txt'
            marker.write_text('preserve')
            with self.assertRaises(FileExistsError):
                cli.prepare_output(root)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_changed_source_fails_before_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'source.py').write_text('changed')
            (root / 'source_lock.json').write_text(json.dumps({'sha256': {'source.py': 'wrong'}}))
            with self.assertRaisesRegex(RuntimeError, 'Frozen E1 source changed'):
                cli.verify_sources(root, root)

    def test_loader_resolves_locked_module_and_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'package').mkdir()
            (root / 'package/__init__.py').write_text('')
            finder = cli.LockedModules(root, {'old_package': 'package/__init__.py'})
            spec = finder.find_spec('old_package')
            self.assertEqual(Path(spec.origin), root / 'package/__init__.py')
            self.assertEqual(spec.submodule_search_locations, [str(root / 'package')])
            self.assertIsNone(finder.find_spec('unrelated'))

    def test_e2_is_not_an_entry_option(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit) as exc:
            cli.main(['replay', '--scene', 'room0', '--out', 'unused', '--variant', 'E2'])
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
