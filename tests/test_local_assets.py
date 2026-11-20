import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from local_assets import ensure_assets


class LocalAssets(unittest.TestCase):
    def fixture(self, root):
        cfg = {'dataset': 'test/data', 'dataset_revision': 'frozen'}
        entries = [{'path': 'data/train.parquet', 'size': 3,
                    'sha256': hashlib.sha256(b'abc').hexdigest(), 'blob': ''}]
        (root / 'configs').mkdir()
        (root / 'configs/assets.json').write_text(json.dumps({'dataset': {
            'repo': cfg['dataset'], 'revision': 'frozen', 'files': entries}}))
        return cfg

    def put(self, path, payload=b'abc'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def test_complete_current_files_never_call_hub(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = self.fixture(root); work = root / 'runtime/base'
            self.put(work / 'dataset/data/train.parquet')
            hub = Mock(side_effect=AssertionError('network must not be called'))
            ensure_assets(cfg, 'dataset', root, work, hub)
            hub.assert_not_called()

    def test_archive_then_other_student_reused_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = self.fixture(root)
            self.put(root / 'runtime_old_20261120/dataset/data/train.parquet')
            hub = Mock(side_effect=AssertionError('network must not be called'))
            for student in ('base', 'small'):
                ensure_assets(cfg, 'dataset', root, root / 'runtime' / student, hub)
                self.assertEqual((root / 'runtime' / student / 'dataset/data/train.parquet').read_bytes(), b'abc')
            hub.assert_not_called()

    def test_corrupt_file_is_downloaded_again(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = self.fixture(root); work = root / 'runtime/base'
            target = work / 'dataset/data/train.parquet'
            self.put(target, b'bad')
            hub = Mock(side_effect=lambda *a, **kw: self.put(target))
            ensure_assets(cfg, 'dataset', root, work, hub)
            self.assertEqual(hub.call_args.kwargs['allow_patterns'], ['data/train.parquet'])

    def test_partial_download_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = self.fixture(root)
            with self.assertRaises(RuntimeError):
                ensure_assets(cfg, 'dataset', root, root / 'runtime/base', Mock())

    def test_wrong_revision_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = self.fixture(root)
            cfg['dataset_revision'] = 'different'
            hub = Mock()
            with self.assertRaises(ValueError):
                ensure_assets(cfg, 'dataset', root, root / 'runtime/base', hub)
            hub.assert_not_called()
