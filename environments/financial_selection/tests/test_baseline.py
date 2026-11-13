"""Controls for the public outcome-blind baseline and reproducible report."""
from copy import deepcopy
import csv
import io
import json
from unittest.mock import patch
import unittest

from financial_selection import baseline
from financial_selection.contract import parse_public
from financial_selection.generator import generate


class BaselineTests(unittest.TestCase):
    def test_markouts_cannot_change_financial_outputs(self):
        for condition in baseline.CONDITIONS:
            public = generate(baseline.PUBLIC_FIXTURE_KEY, "dev", 7, condition=condition).public_files
            changed = deepcopy(public)
            rows = list(csv.DictReader(io.StringIO(changed["observed_labels.csv"])))
            replacements = {}
            for index, row in enumerate(rows):
                replacements.setdefault(row["release_id"], str((-1 if index % 2 else 1) * (98765 + index)))
                row["markout_bp"] = replacements[row["release_id"]]
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            changed["observed_labels.csv"] = output.getvalue()
            parse_public(changed)
            original_policy, changed_policy = baseline.policy(public), baseline.policy(changed)
            self.assertNotEqual(original_policy.pop("input_sha256"), changed_policy.pop("input_sha256"))
            self.assertEqual(original_policy, changed_policy)
            self.assertEqual(changed_policy["allocation"], [0] * len(changed_policy["baskets"]))

    def test_public_hundred_fixture_baseline_and_stdout_only_entrypoint(self):
        report = baseline.report()
        self.assertEqual(sum(row["n"] for row in report["conditions"].values()), 100)
        self.assertAlmostEqual(report["overall_mean_reward"], 0.6745416666666667, places=14)
        self.assertEqual(report["overall_full_policy_success_count"], 0)
        self.assertEqual(report["overall_full_policy_success_rate"], 0)
        self.assertEqual([report["conditions"][condition]["n"] for condition in baseline.CONDITIONS], [20] * 5)
        with patch.object(baseline, "report", return_value=report), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            baseline.main()
        self.assertEqual(json.loads(stdout.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
