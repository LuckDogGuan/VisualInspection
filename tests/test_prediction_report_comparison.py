import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification.compare_prediction_reports import compare_records, load_prediction_records


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["source_name", "source_path", "status", "class_id", "confidence"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PredictionReportComparisonTests(unittest.TestCase):
    def test_load_prediction_records_uses_source_name_and_filename_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "predictions.csv"
            write_csv(
                csv_path,
                [
                    {"source_name": "APSPC1", "source_path": "/a/APSPC1/img0.jpg", "status": "ok", "class_id": "0", "confidence": "0.9"},
                    {"source_name": "APSPC2", "source_path": "/b/APSPC2/img0.jpg", "status": "ok", "class_id": "5", "confidence": "0.8"},
                ],
            )

            records = load_prediction_records(csv_path)

        self.assertEqual(sorted(records), ["APSPC1/img0.jpg", "APSPC2/img0.jpg"])

    def test_compare_records_reports_class_changes_and_confidence_delta(self):
        baseline = {
            "APSPC1/img0.jpg": {"status": "ok", "class_id": "0", "confidence": "0.900"},
            "APSPC1/img1.jpg": {"status": "ok", "class_id": "5", "confidence": "0.800"},
        }
        current = {
            "APSPC1/img0.jpg": {"status": "ok", "class_id": "0", "confidence": "0.700"},
            "APSPC1/img1.jpg": {"status": "ok", "class_id": "10", "confidence": "0.600"},
            "APSPC1/img2.jpg": {"status": "ok", "class_id": "1", "confidence": "0.500"},
        }

        summary = compare_records(baseline, current, confidence_tolerance=0.05)

        self.assertEqual(summary["common_count"], 2)
        self.assertEqual(summary["missing_count"], 0)
        self.assertEqual(summary["added_count"], 1)
        self.assertEqual(summary["class_changed_count"], 1)
        self.assertEqual(summary["confidence_changed_count"], 2)
        self.assertAlmostEqual(summary["max_confidence_delta"], 0.2)


if __name__ == "__main__":
    unittest.main()
