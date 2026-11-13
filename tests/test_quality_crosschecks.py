"""Reproducible alternate methods and their public-input isolation boundary."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from quality.crosschecks import methods, runner


ROOT = Path(__file__).resolve().parents[1]


def question(case_id):
    packet = json.loads((ROOT / runner.CASES[case_id][0]).read_text())
    return packet["question"] if case_id.startswith("c") else packet


class AlternateMethods(unittest.TestCase):
    def test_exhaustive_stopping_counts_first_crossing_not_terminal_exceedance(self):
        answers, certificate = methods.q098(question("q098"))
        self.assertEqual(certificate["complete_paths"], 262144)
        self.assertEqual(certificate["first_crossing_paths"], 2786)
        self.assertGreater(certificate["first_crossing_paths"], certificate["terminal_exceedance_paths"])
        self.assertEqual(answers["finite_horizon_false_promotion"], float(Fraction(2786, 262144)))
        self.assertLessEqual(answers["finite_horizon_false_promotion"], answers["anytime_false_promotion_bound"])
        self.assertGreater(answers["likelihood_stop_day"], answers["naive_fixed_test_stop_day"])

    def test_unreviewed_stopping_domain_is_explicitly_rejected(self):
        packet = question("q098")
        packet["inputs"]["horizon_days"] = 19
        with self.assertRaisesRegex(ValueError, "reviewed fair-null 18-day"):
            methods.q098(packet)

    def test_complete_collateral_grid_preserves_cash_and_eligibility_equality(self):
        packet = question("q033")
        answers, certificate = methods.q033(packet)
        self.assertEqual(certificate["complete_grid_points"], 7 * 6 * 501)
        self.assertEqual(certificate["recognized_surplus_exact"], "0")
        self.assertEqual(answers["cash"], .72)
        packet["inputs"]["availability"][2] = .72
        self.assertEqual(methods.q033(packet)[0], answers)
        packet["inputs"]["availability"][2] = .71
        with self.assertRaisesRegex(ValueError, "No feasible collateral"):
            methods.q033(packet)

    def test_collateral_cash_rounds_up_to_stated_grid_and_corporate_limit_binds(self):
        packet = question("q033")
        packet["inputs"]["required_recognized"] = 10.001
        answers, _ = methods.q033(packet)
        self.assertEqual(answers["cash"], .73)
        self.assertEqual(answers["recognized_value"], 10.01)
        packet = question("q033")
        packet["inputs"]["corporate_recognized_cap"] = 3.4
        self.assertEqual(methods.q033(packet)[0]["bond_blocks"][1], 4)
        packet["inputs"]["corporate_recognized_cap"] = 3.399
        self.assertLess(methods.q033(packet)[0]["bond_blocks"][1], 4)

    def test_analytic_caplet_forward_derivative_and_notional_scaling(self):
        packet = question("q065")
        answers, certificate = methods.q065(packet)
        step = 1e-7
        up, down = deepcopy(packet), deepcopy(packet)
        up["inputs"]["forward"] += step
        down["inputs"]["forward"] -= step
        derivative = (methods.q065(up)[0]["normal_price"] - methods.q065(down)[0]["normal_price"]) / (2 * step)
        self.assertAlmostEqual(derivative / 10000, answers["normal_forward_pv01"], places=7)
        self.assertGreater(certificate["normal_exercise_probability"], 0)
        self.assertLess(certificate["normal_exercise_probability"], 1)
        self.assertFalse(answers["unshifted_black_admissible"])
        packet["inputs"]["notional"] *= 2
        scaled, _ = methods.q065(packet)
        for key in ("normal_price", "shifted_black_price", "normal_forward_pv01"):
            self.assertAlmostEqual(scaled[key], 2 * answers[key])

    def test_core_trade_search_counts_every_optimum_and_both_cash_checkpoints(self):
        packet = question("c201")
        answers, certificate = methods.c201(packet)
        self.assertEqual(certificate["complete_trade_vectors"], 2401)
        self.assertEqual(answers["optimal_plan_count"], 2)
        self.assertEqual(len(certificate["all_optimal_holdings"]), 2)
        self.assertEqual(answers["final_holdings"], min(certificate["all_optimal_holdings"]))
        for holdings in certificate["all_optimal_holdings"]:
            trades = [h - o for h, o in zip(holdings, packet["inputs"]["opening_holdings"])]
            cost = sum(1 + Fraction(q * q, 4) for q in trades if q)
            # Final self-financing cash identity is separate from the early ledger.
            final = 20 - 35 - 10 * sum(trades) - cost
            late = 10 * sum(max(-trades[j], 0) for j in (1, 3))
            self.assertEqual(float(cost), answers["cost"])
            self.assertGreaterEqual(final, 5)
            self.assertGreaterEqual(final - late, 0)

    def test_shared_sign_protocol_and_regime_gate(self):
        answers, certificate = methods.c214(question("c214"))
        self.assertEqual(answers["eligible_blocks"], ["b1", "b2", "b4", "b5", "b7", "b8"])
        self.assertEqual((answers["selected_policy"], answers["selected_checkpoint"]), ("A", 6))
        self.assertEqual((certificate["raw_exceedances"], certificate["selection_exceedances"]), (2, 4))
        self.assertEqual(answers["unadjusted_p"], 2 / 64)
        self.assertEqual(answers["selection_adjusted_p"], 4 / 64)
        self.assertTrue(certificate["decimal_ordering_agrees_with_exact_cross_products"])
        self.assertLess(Fraction(certificate["late_regime_mean_exact"]), Fraction(certificate["early_regime_mean_exact"]) / 2)
        self.assertFalse(answers["deploy"])

    def test_exact_score_ordering_preserves_equalities_signs_and_tiny_gaps(self):
        left = methods._score([Fraction(1), Fraction(1)])
        equal = methods._score([Fraction(2), Fraction(2)])
        almost = methods._score([Fraction(1), Fraction(1) + Fraction(1, 10 ** 100)])
        self.assertEqual(methods._compare_scores(left, equal), 0)
        self.assertEqual(methods._decimal_score(left), methods._decimal_score(almost))
        self.assertGreater(methods._compare_scores(left, almost), 0)
        self.assertLess(methods._compare_scores((-left[0], left[1]), (-almost[0], almost[1])), 0)
        self.assertGreater(methods._compare_scores((Fraction(0), Fraction(1)), (-left[0], left[1])), 0)

    def test_core_selection_ties_use_alphabetical_policy_then_longer_checkpoint(self):
        packet = question("c214")
        tape = next(source for source in packet["sources"] if source["id"] == "qt_tape")
        lines = tape["text"].splitlines()
        updated = [lines[0]]
        eligible = 0
        for line in lines[1:]:
            parts = [part.strip() for part in line.split("|")]
            decision, publication, training, market = map(int, parts[1:5])
            if publication <= decision and training + 2 <= decision:
                # Three ones followed by three zeros give exactly equal scores
                # at checkpoints3 and6, identically for all three policies.
                net = 1 if eligible < 3 else 0
                eligible += 1
                parts[5] = f"{net + 1 + market},{net + 2.5},{net + 1 - market}"
            updated.append(" | ".join(parts))
        tape["text"] = "\n".join(updated)
        answers, _ = methods.c214(packet)
        self.assertEqual((answers["selected_policy"], answers["selected_checkpoint"]), ("A", 6))
        self.assertEqual(answers["selection_adjusted_p"], 1 / 8)
        self.assertEqual(answers["late_regime_mean"], 0)
        self.assertFalse(answers["deploy"])


class CrosscheckBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = runner.run()

    def test_all_five_cases_and_28_fields_agree_with_primary_arithmetic(self):
        self.assertEqual(self.report["status"], "agrees_with_primary_audit")
        self.assertEqual((self.report["case_count"], self.report["output_count"]), (5, 28))
        self.assertTrue(all(field["matches"] for row in self.report["records"]
                            for field in row["primary_comparison"]["fields"]))
        self.assertEqual(set(self.report["implementation_sha256"]), {
            f"quality/crosschecks/{name}" for name in
            ("__init__.py", "methods.py", "runner.py", "__main__.py", "reviewed_inputs.json")
        })

    def test_packet_drift_rejected_before_primary_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, _ in runner.CASES.values():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            changed = root / runner.CASES["c214"][0]
            changed.write_bytes(changed.read_bytes() + b"\n")
            with patch.object(runner, "_primary") as primary:
                with self.assertRaisesRegex(runner.InputDriftError, "reviewed hash"):
                    runner.run(root)
                primary.assert_not_called()

    def test_every_alternate_derivation_precedes_first_primary_call(self):
        events = []
        originals = runner.CHECKS.copy()
        primary = runner._primary

        def alternate(case_id):
            def calculate(packet):
                result = originals[case_id](packet)
                events.append(case_id)
                return result
            return calculate

        def compare(case_id, packet):
            self.assertEqual(set(events), set(runner.CASES))
            return primary(case_id, packet)

        with patch.dict(runner.CHECKS, {cid: alternate(cid) for cid in runner.CASES}):
            with patch.object(runner, "_primary", compare):
                self.assertEqual(runner.run()["status"], "agrees_with_primary_audit")

    def test_primary_corruption_is_reported_without_changing_alternate_derivation(self):
        original = runner._primary

        def corrupted(case_id, packet):
            answers = original(case_id, packet)
            if case_id == "q033":
                answers = dict(answers, cash=answers["cash"] + .5)
            return answers

        with patch.object(runner, "_primary", corrupted):
            changed = runner.run()
        self.assertEqual(changed["status"], "primary_audit_disagreement")
        baseline = {row["case_id"]: row["alternate_answers"] for row in self.report["records"]}
        for row in changed["records"]:
            self.assertEqual(row["alternate_answers"], baseline[row["case_id"]])
        mismatches = [(row["case_id"], field["field"]) for row in changed["records"]
                      for field in row["primary_comparison"]["fields"] if not field["matches"]]
        self.assertEqual(mismatches, [("q033", "cash")])

    def test_derive_only_imports_no_primary_inventory_or_scorer_modules(self):
        code = """
import json,sys
from quality.crosschecks.runner import run
report=run(compare_primary=False)
for name in sys.modules:
    assert not name.startswith(('scoring_v2','scoring_v3','private_evaluator'))
    assert name not in ('quality.inventory','quality.numerical','quality.risk_numerical','quality.core_numerical')
assert all('primary_comparison' not in row for row in report['records'])
print(json.dumps({'status':report['status'],'cases':report['case_count']}))
"""
        process = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(process.stdout), {"status": "derived_only", "cases": 5})

    def test_cli_is_reproducible_and_emits_only_relative_packet_paths(self):
        command = [sys.executable, "-B", "-m", "quality.crosschecks", "--check"]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        summary = json.loads(process.stdout)
        self.assertEqual(summary["status"], "agrees_with_primary_audit")
        self.assertEqual(summary["output_count"], 28)
        encoded = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertNotIn(str(ROOT), encoded)
        self.assertNotIn("stored_reference", encoded)
        for row in self.report["records"]:
            self.assertFalse(Path(row["input"]["path"]).is_absolute())
            self.assertEqual(len(row["input"]["sha256"]), 64)

    def test_comparison_does_not_coerce_booleans_or_accept_nonfinite_numbers(self):
        self.assertFalse(runner._equal(True, 1, 1e-6, 0))
        self.assertFalse(runner._equal([1], [True], 1e-6, 0))
        self.assertFalse(runner._equal(math.nan, math.nan, 0, 0))
        self.assertFalse(runner._equal(math.inf, math.inf, 0, 0))
        self.assertFalse(runner._equal(10 ** 400, 1, 0, 0))
        self.assertTrue(runner._equal(1.0, 1, 0, 0))


if __name__ == "__main__":
    unittest.main()
