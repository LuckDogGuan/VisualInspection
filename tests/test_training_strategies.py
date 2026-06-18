import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification.data import LabelRow
from classification.train import FocalLoss, build_class_weights, build_weighted_sampler


class TrainingStrategyTests(unittest.TestCase):
    def test_build_class_weights_gives_rare_classes_larger_weights(self):
        rows = [
            LabelRow(Path("normal1.jpg"), 0, "normal"),
            LabelRow(Path("normal2.jpg"), 0, "normal"),
            LabelRow(Path("normal3.jpg"), 0, "normal"),
            LabelRow(Path("defect.jpg"), 1, "defect"),
        ]

        weights = build_class_weights(rows, num_classes=2)

        self.assertEqual(tuple(weights.shape), (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_build_weighted_sampler_assigns_rare_rows_larger_sample_weight(self):
        rows = [
            LabelRow(Path("normal1.jpg"), 0, "normal"),
            LabelRow(Path("normal2.jpg"), 0, "normal"),
            LabelRow(Path("defect.jpg"), 1, "defect"),
        ]

        sampler = build_weighted_sampler(rows)

        self.assertEqual(sampler.num_samples, 3)
        self.assertLess(float(sampler.weights[0]), float(sampler.weights[2]))

    def test_focal_loss_downweights_easy_examples(self):
        logits = torch.tensor([[5.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
        targets = torch.tensor([0, 0], dtype=torch.long)
        focal = FocalLoss(gamma=2.0, reduction="none")
        ce = torch.nn.CrossEntropyLoss(reduction="none")

        focal_losses = focal(logits, targets)
        ce_losses = ce(logits, targets)

        self.assertLess(float(focal_losses[0]), float(ce_losses[0]))
        self.assertLess(float(focal_losses[0]), float(focal_losses[1]))


if __name__ == "__main__":
    unittest.main()
