import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.utils import load_hf_dataset


class LocalTOFUDataTest(unittest.TestCase):
    def test_local_override_loads_named_jsonl_split(self):
        records = [
            {"question": "Question one?", "answer": "Answer one."},
            {"question": "Question two?", "answer": "Answer two."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forget10.json"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "TOFU_LOCAL_DIR": directory,
                    "TOFU_LOCAL_CACHE_DIR": str(Path(directory) / "cache"),
                },
            ):
                dataset = load_hf_dataset(
                    "locuslab/TOFU", name="forget10", split="train"
                )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["question"], "Question one?")

    def test_local_override_reports_missing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TOFU_LOCAL_DIR": directory}):
                with self.assertRaises(FileNotFoundError):
                    load_hf_dataset(
                        "locuslab/TOFU", name="forget10", split="train"
                    )


if __name__ == "__main__":
    unittest.main()
