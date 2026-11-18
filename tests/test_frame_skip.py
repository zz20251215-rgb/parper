import unittest

from scripts.calibrate_frame_skip import sweep


class FrameSkipCalibrationTests(unittest.TestCase):
    def test_sweep_reports_validation_tradeoff(self):
        rows = [
            {'positive': True, 'score': 0.9},
            {'positive': True, 'score': 0.4},
            {'positive': False, 'score': 0.8},
            {'positive': False, 'score': 0.1},
        ]
        result = sweep(rows, 'score')
        by_threshold = {row['threshold']: row for row in result}
        self.assertEqual(by_threshold[0.8]['positive_recall'], 0.5)
        self.assertEqual(by_threshold[0.8]['negative_false_positive_rate'], 0.5)
        self.assertEqual(by_threshold[0.9]['positive_recall'], 0.5)
        self.assertEqual(by_threshold[0.9]['negative_false_positive_rate'], 0.0)

    def test_sweep_handles_no_negative_frames(self):
        rows = [
            {'positive': True, 'score': 0.2},
            {'positive': True, 'score': 0.7},
        ]
        result = sweep(rows, 'score')
        self.assertTrue(all(row['negative_false_positive_rate'] is None for row in result))
        threshold = next(row for row in result if row['threshold'] == 0.7)
        self.assertEqual(threshold['positive_recall'], 0.5)


if __name__ == '__main__':
    unittest.main()
