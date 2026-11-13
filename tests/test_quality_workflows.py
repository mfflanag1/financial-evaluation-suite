"""Financial identities and information boundaries for independent workflow checks."""
import copy
import json
from pathlib import Path
import unittest

from quality import workflow_numerical as audit


ROOT = Path(__file__).resolve().parents[1]


def packet(number):
    return json.loads((ROOT / "questions" / "legacy" / "workflows" / "questions" / f"q{number}.json").read_text())


class WorkflowNumericalTests(unittest.TestCase):
    def test_all_public_outputs_are_covered_and_row_order_is_irrelevant(self):
        for number in range(101, 107):
            original = packet(number)
            shuffled = copy.deepcopy(original)
            for key, values in shuffled["inputs"].items():
                if isinstance(values, list) and values and isinstance(values[0], dict) and key not in ("formations", "industries"):
                    values.reverse()
            with self.subTest(case=number):
                first = audit.audit_case(original)
                second = audit.audit_case(shuffled)
                self.assertEqual(first["answers"], second["answers"])
                self.assertEqual(first["unsupported"], [])
                self.assertEqual(first["ambiguities"], [])

    def test_multicurrency_nav_reconciles_and_scales_without_return_change(self):
        original = packet(101)
        baseline = audit.audit_case(original)
        self.assertEqual(baseline["certificate"]["exact_nav_bridge_residuals"], [0, 0])
        scaled = copy.deepcopy(original)
        for key, fields in {"opening_positions": ("quantity",), "opening_cash": ("amount",),
                            "external_flows": ("amount",), "fills": ("quantity", "fee"),
                            "cash_expenses": ("amount",)}.items():
            for row in scaled["inputs"][key]:
                for field in fields:
                    row[field] *= 3
        changed = audit.audit_case(scaled)["answers"]
        for key in ("daily_returns", "time_weighted_return"):
            self.assertEqual(changed[key], baseline["answers"][key])
        for actual, expected in zip(changed["daily_nav_usd"], baseline["answers"]["daily_nav_usd"]):
            self.assertAlmostEqual(actual, 3 * expected)

    def test_settlement_changes_liquidity_without_changing_trade_date_nav(self):
        original = packet(102)
        baseline = audit.audit_case(original)
        self.assertEqual(baseline["certificate"]["exact_nav_bridge_residuals"], [0, 0, 0])
        delayed = copy.deepcopy(original)
        for row in delayed["inputs"]["trades"]:
            row["settlement_date"] = "2025-02-01"
        changed = audit.audit_case(delayed)["answers"]
        self.assertEqual(changed["daily_nav_usd"], baseline["answers"]["daily_nav_usd"])
        self.assertEqual(changed["daily_pnl_bridge_usd"], baseline["answers"]["daily_pnl_bridge_usd"])
        self.assertNotEqual(changed["daily_balance_sheet_usd"], baseline["answers"]["daily_balance_sheet_usd"])

    def test_split_preserves_basis_and_commission_is_not_fifo_basis(self):
        original = packet(102)
        baseline = audit.audit_case(original)["answers"]
        changed = copy.deepcopy(original)
        for row in changed["inputs"]["trades"]:
            row["fee"] += 1
        result = audit.audit_case(changed)["answers"]
        self.assertEqual(result["ending_security_ledger"], baseline["ending_security_ledger"])
        self.assertEqual(result["daily_pnl_bridge_usd"][::4], baseline["daily_pnl_bridge_usd"][::4])
        self.assertAlmostEqual(result["daily_nav_usd"][-1], baseline["daily_nav_usd"][-1] - 6)

    def test_fee_equation_accounts_for_postfee_target(self):
        fee, turnover = audit.fee_fixed_point([0, 0], [0.5, -0.5], 0.01)
        self.assertAlmostEqual(fee, 0.01 / 1.01)
        self.assertAlmostEqual(turnover, 1 / 1.01)
        self.assertAlmostEqual(fee, 0.01 * turnover)

    def test_signal_holdout_does_not_select_lookback(self):
        original = packet(103)
        baseline = audit.audit_case(original)["answers"]
        altered = copy.deepcopy(original)
        for row in altered["inputs"]["rows"]:
            if row["month"] >= altered["inputs"]["test_start"]:
                row["returns"] = [value * -0.7 for value in row["returns"]]
        result = audit.audit_case(altered)["answers"]
        for key in ("validation_sharpes", "chosen_lookback_months"):
            self.assertEqual(result[key], baseline[key])
        self.assertNotEqual(result["test_net_returns"], baseline["test_net_returns"])

    def test_future_filings_cannot_change_past_portfolios(self):
        original = packet(104)
        baseline = audit.audit_case(original)
        altered = copy.deepcopy(original)
        for row in altered["inputs"]["filings"]:
            if row["accepted_at"] > altered["inputs"]["formations"][-1]["cutoff"]:
                row["ttm_earnings"] *= 1000
        self.assertEqual(audit.audit_case(altered), baseline)

    def test_covariance_holdout_does_not_enter_validation_or_final_weights(self):
        original = packet(105)
        baseline = audit.audit_case(original)
        self.assertLess(abs(baseline["certificate"]["weight_sum_residual"]), 1e-12)
        self.assertLess(baseline["certificate"]["marginal_variance_spread"], 1e-12)
        altered = copy.deepcopy(original)
        for row in altered["inputs"]["rows"]:
            if row["month"] >= altered["inputs"]["test_window"]["start"]:
                row["returns"] = [2 * value for value in row["returns"]]
        result = audit.audit_case(altered)["answers"]
        for key in ("selected_shrinkage", "validation_variance_by_candidate", "final_weights"):
            self.assertEqual(result[key], baseline["answers"][key])
        self.assertAlmostEqual(result["test_annualized_volatility"], 2 * baseline["answers"]["test_annualized_volatility"])

    def test_ridge_holdout_does_not_select_or_refit_model(self):
        original = packet(106)
        baseline = audit.audit_case(original)
        self.assertLess(max(baseline["certificate"]["refit_normal_equation_residuals"]), 1e-11)
        altered = copy.deepcopy(original)
        end = original["inputs"]["split"]["train"] + original["inputs"]["split"]["validation"]
        test_periods = {row["t"] for row in sorted(original["inputs"]["rows"], key=lambda row: row["t"])[end:]}
        for row in altered["inputs"]["rows"]:
            if row["t"] in test_periods:
                row["y"] += 2
        result = audit.audit_case(altered)["answers"]
        for key in ("selected_lambda", "validation_mse_by_lambda", "final_coefficients"):
            self.assertEqual(result[key], baseline["answers"][key])
        self.assertNotEqual(result["test_mse_by_lambda"], baseline["answers"]["test_mse_by_lambda"])


if __name__ == "__main__":
    unittest.main()
