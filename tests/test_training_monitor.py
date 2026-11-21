import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training_monitor import records, rewind_log, draw_curves, monitor_update_allowed, source_map_matches


class Monitor(unittest.TestCase):
    def test_source_map_accepts_only_known_line_endings_and_path_separators(self):
        expected = {'src/engine.py': ['LF-digest', 'CRLF-digest']}
        self.assertTrue(source_map_matches({'src\\engine.py': 'CRLF-digest'}, expected))
        self.assertTrue(source_map_matches({'src/engine.py': 'LF-digest'}, expected))
        self.assertFalse(source_map_matches({'src/engine.py': 'edited'}, expected))
        self.assertFalse(source_map_matches({'src/engine.py': 'LF-digest', 'src/extra.py': 'x'}, expected))
    def test_resume_rewinds_partial_epoch_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            original = '\n'.join(json.dumps({'epoch': e, 'sample': s})
                                 for e, s in [(1,1), (1,2), (2,1), (2,1)]) + '\n{"epoch":'
            (out / 'train.jsonl').write_text(original)
            rewind_log(out, 1)
            self.assertEqual([(r['epoch'],r['sample']) for r in records(out / 'train.jsonl')], [(1,1),(1,2)])
            self.assertEqual(next((out / 'interrupted_logs').glob('*.jsonl')).read_text(), original)

    def test_restart_before_first_checkpoint_rewinds_all(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'train.jsonl').write_text(json.dumps({'epoch':1,'sample':1})+'\n')
            rewind_log(out, 0)
            self.assertEqual(records(out / 'train.jsonl'), [])

    def test_curves_are_valid_svg_for_legacy_and_new_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            row = {'epoch':1,'sample':1,'loss':1.2,'ce':1,'kd':2,'adv':0,'d':0}
            (out / 'train.jsonl').write_text(json.dumps(row)+'\n')
            (out / 'dev_epoch_1.json').write_text(json.dumps({'wer':.1,'rtf':.02}))
            draw_curves(out)
            ElementTree.parse(out / 'training_curves.svg')
            text = (out / 'training_curves.svg').read_text(encoding='utf-8')
            for title in ['Total loss', 'CE loss', 'KD loss', 'Dev WER', 'Dev RTF']:
                self.assertIn(title, text)
            self.assertTrue((out / 'training_curves.html').exists())

    def test_monitor_update_requires_both_exact_source_maps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'configs').mkdir()
            (root / 'configs/monitor_compatibility.json').write_text(json.dumps({
                'previous':[{'engine':'old'}], 'current':{'engine':'new'}}))
            self.assertTrue(monitor_update_allowed(root, {'engine':'old'}, {'engine':'new'}))
            self.assertFalse(monitor_update_allowed(root, {'engine':'other'}, {'engine':'new'}))
            self.assertFalse(monitor_update_allowed(root, {'engine':'old'}, {'engine':'edited'}))
