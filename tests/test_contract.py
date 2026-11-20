import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import alignment, normalize, evaluation_path

class Contract(unittest.TestCase):
    def test_adversarial_outputs_do_not_collide(self):
        for suffix in ('.json', '.jsonl'):
            outputs = [evaluation_path(Path('evaluation') / name, suffix)
                       for name in ('kd_adv_0.01', 'kd_adv_0.1')]
            self.assertEqual(len(set(outputs)), 2)
            self.assertEqual(outputs[0].name, 'kd_adv_0.01' + suffix)
            self.assertEqual(outputs[1].name, 'kd_adv_0.1' + suffix)

    def test_published_teacher_no_speech_alias(self):
        self.assertEqual(alignment({'<|nocaptions|>': 0, 'eos': 1},
                                   {'<|nospeech|>': 2, 'eos': 1}), [2, 1])

    def test_vocab_permutation_and_extra_teacher_token(self):
        self.assertEqual(alignment({'a': 0, 'eos': 1}, {'extra': 0, 'eos': 1, 'a': 2}), [2, 1])

    def test_missing_token_fails(self):
        with self.assertRaises(ValueError): alignment({'a': 0}, {'b': 0})

    def test_normalization(self):
        self.assertEqual(normalize("Oscar-Kilo, DON'T 123."), 'oscar kilo dont 123')

if __name__ == '__main__': unittest.main()
