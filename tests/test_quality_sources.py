"""Retained-source arithmetic, vintage boundaries and scenario invariants."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from quality.source_numerical import CHECKS

ROOT = Path(__file__).resolve().parents[1]


def packet(cid):
    folder = ROOT / "questions/legacy/source_grounded/cases" / cid
    question = json.loads((folder / "case.json").read_text())
    sources = {sid: (folder / "sources" / f"{sid}.txt").read_text() for sid in question["source_ids"]}
    return question, sources


class SourceCalculations(unittest.TestCase):
    def test_all_cases_are_derived_before_reference_comparison(self):
        derived = {cid: calculate(*packet(cid))[0] for cid, calculate in CHECKS.items()}
        for cid, answers in derived.items():
            question, _ = packet(cid)
            self.assertEqual(set(answers), {field["id"] for field in question["output_fields"]})
            gold = json.loads((ROOT / "private_evaluator/legacy/source_grounded/gold" / f"{cid}.json").read_text())
            for field in gold["fields"]:
                actual, expected = answers[field["id"]], field["expected"]
                with self.subTest(case=cid, field=field["id"]):
                    if isinstance(expected, (bool, str)):
                        self.assertIs(type(actual), type(expected))
                        self.assertEqual(actual, expected)
                    else:
                        self.assertTrue(math.isfinite(actual))
                        self.assertLessEqual(abs(actual - expected), field["atol"] + field["rtol"] * abs(expected))

    def test_payroll_new_month_changes_coverage_but_not_prior_revision(self):
        question, sources = packet("macro-001")
        initial = CHECKS["macro-001"](question, sources)[0]
        sources["bls-20240405"] = sources["bls-20240405"].replace("rose by 303,000", "rose by 304,000")
        changed = CHECKS["macro-001"](question, sources)[0]
        for field in ("mar_first_print", "q1_first_print_sum", "q1_asof_apr05_sum", "apr05_increment_to_tracked_q1_change"):
            self.assertEqual(changed[field] - initial[field], 1)
        for field in ("apr05_prior_two_month_revision", "q1_cumulative_revision_vs_first_prints", "mar08_prior_two_month_revision"):
            self.assertEqual(changed[field], initial[field])

    def test_conflicting_repeated_payroll_facts_fail_closed(self):
        question, sources = packet("macro-001")
        sources["bls-20240405"] = sources["bls-20240405"].replace("rose by 303,000", "rose by 304,000", 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            CHECKS["macro-001"](question, sources)

    def test_payroll_source_precision_is_not_silently_truncated(self):
        question, sources = packet("macro-001")
        sources["bls-20240405"] = sources["bls-20240405"].replace("rose by 303,000", "rose by 303,500")
        with self.assertRaisesRegex(ValueError, "exact integer thousand"):
            CHECKS["macro-001"](question, sources)

    def test_reported_payroll_revision_must_match_operand_difference(self):
        question, sources = packet("macro-001")
        sources["bls-20240308"] = sources["bls-20240308"].replace("down by 124,000", "down by 125,000")
        with self.assertRaisesRegex(ValueError, "revision does not reconcile"):
            CHECKS["macro-001"](question, sources)

    def test_cpi_expectation_changes_surprise_only(self):
        question, sources = packet("macro-002")
        initial = CHECKS["macro-002"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("SYNTHETIC: 0.2%", "SYNTHETIC: 0.25%")
        changed = CHECKS["macro-002"](question, sources)[0]
        self.assertAlmostEqual(changed.pop("january_surprise_given_expectation"), .05)
        initial.pop("january_surprise_given_expectation")
        self.assertEqual(changed, initial)

    def test_gdp_retains_rounding_residual_and_nonlinear_conversion(self):
        values = CHECKS["macro-003"](*packet("macro-003"))[0]
        self.assertAlmostEqual(values["headline_minus_component_revision_gap"], .01)
        self.assertAlmostEqual((1 + values["third_implied_quarterly_growth"] / 100)**4, 1.021)
        self.assertNotAlmostEqual(values["third_implied_quarterly_growth"], 2.1 / 4)
        self.assertFalse(values["cross_base_level_ratio_is_real_growth"])

    def test_gas_incremental_demand_changes_scenario_not_observed_flow(self):
        question, sources = packet("energy-001")
        initial = CHECKS["energy-001"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("with 7 Bcf/day", "with 8 Bcf/day")
        changed = CHECKS["energy-001"](question, sources)[0]
        self.assertEqual(changed["scenario_terminal_working_stock"] - initial["scenario_terminal_working_stock"], -14)
        self.assertEqual(changed["two_week_implied_flow"], initial["two_week_implied_flow"])
        self.assertEqual(changed["scenario_max_extra_demand"], initial["scenario_max_extra_demand"])

    def test_gas_historical_survey_interval_is_not_future_scenario_horizon(self):
        question, sources = packet("energy-001")
        initial = CHECKS["energy-001"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("Over the next 14 days", "Over the next 28 days")
        changed = CHECKS["energy-001"](question, sources)[0]
        self.assertEqual(changed["two_week_average_implied_flow"], initial["two_week_average_implied_flow"])
        self.assertAlmostEqual(changed["scenario_max_extra_demand"] * 2, initial["scenario_max_extra_demand"])

    def test_petroleum_reduction_moves_commercial_stock_one_for_one(self):
        question, sources = packet("energy-002")
        initial = CHECKS["energy-002"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("reduce crude imports by 400", "reduce crude imports by 500")
        changed = CHECKS["energy-002"](question, sources)[0]
        self.assertEqual(changed["scenario_commercial_stock_change"] - initial["scenario_commercial_stock_change"], -100)
        self.assertAlmostEqual(changed["scenario_weekly_commercial_build"] - initial["scenario_weekly_commercial_build"], -.7)
        self.assertEqual(changed["maximum_import_reduction_without_commercial_draw"], initial["maximum_import_reduction_without_commercial_draw"])
        self.assertEqual(changed["flow_minus_stock_rounding_gap"], 2)
        self.assertEqual(changed["net_transfer_across_crude_and_products"], 0)

    def test_margin_consumes_liquidity_without_changing_terminal_economics(self):
        question, sources = packet("portfolio-002")
        initial, certificate = CHECKS["portfolio-002"](question, sources)
        question["prompt"] = question["prompt"].replace("fee 150, margin 2500", "fee 150, margin 2550")
        changed, revised = CHECKS["portfolio-002"](question, sources)
        self.assertEqual(changed["optimal_minimum_cash"], 0)
        self.assertEqual(changed["optimal_plan"], initial["optimal_plan"])
        self.assertEqual(changed["optimal_worst_pnl"], initial["optimal_worst_pnl"])
        for plan, original in certificate["operands"]["plan_certificates"].items():
            self.assertEqual(original["pnl"], revised["operands"]["plan_certificates"][plan]["pnl"])
        question["prompt"] = question["prompt"].replace("margin 2550", "margin 2551")
        beyond = CHECKS["portfolio-002"](question, sources)[0]
        self.assertEqual(beyond["optimal_plan"], "B")
        self.assertNotIn("C", beyond["feasible_plans"].split("|"))
        self.assertEqual(beyond["first_optimizer_change_budget"], 7001)

    def test_cash_one_dollar_before_and_at_optimizer_entry(self):
        question, sources = packet("portfolio-002")
        question["prompt"] = question["prompt"].replace("Starting free cash is 7000", "Starting free cash is 6949")
        before = CHECKS["portfolio-002"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("Starting free cash is 6949", "Starting free cash is 6950")
        at = CHECKS["portfolio-002"](question, sources)[0]
        self.assertEqual(before["optimal_plan"], "B")
        self.assertEqual(before["first_optimizer_change_budget"], 6950)
        self.assertEqual(at["optimal_plan"], "C")
        self.assertEqual(at["optimal_minimum_cash"], 0)

    def test_missing_margin_spans_three_optimizer_regimes(self):
        answers, certificate = CHECKS["portfolio-002"](*packet("portfolio-002"))
        self.assertEqual(answers["combined_parallel_dv01"], 0)
        self.assertNotEqual(answers["base_scenario_pnl"], "H=0;R=0;T=0")
        self.assertEqual(certificate["operands"]["missing_margin_optimizer_sets"], [["B"], ["BC"], ["C"]])
        self.assertEqual(answers["missing_margin_optimizer_status"], "insufficient_data")

    def test_one_second_latency_change_removes_fixed_snapshot_leakage_gap(self):
        question, sources = packet("systematic-001")
        initial = CHECKS["systematic-001"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("processing another 60 seconds", "processing another 59 seconds")
        changed = CHECKS["systematic-001"](question, sources)[0]
        self.assertEqual(changed["before_ready_selected_asset"], "nvidia")
        self.assertEqual(changed["leakage_net_profit_inflation"], 0)
        self.assertAlmostEqual(changed["valid_toy_net_profit"] - initial["valid_toy_net_profit"], initial["leakage_net_profit_inflation"])
        self.assertEqual(changed["nvidia_public_eligibility_at_acceptance"], "not_established")

    def test_cost_increase_uses_actual_exit_values(self):
        question, sources = packet("systematic-001")
        initial = CHECKS["systematic-001"](question, sources)[0]
        question["prompt"] = question["prompt"].replace("Charge 5 bps", "Charge 6 bps")
        changed = CHECKS["systematic-001"](question, sources)[0]
        self.assertAlmostEqual(changed["valid_toy_net_profit"] - initial["valid_toy_net_profit"], -3.99)
        self.assertAlmostEqual(changed["leakage_net_profit_inflation"] - initial["leakage_net_profit_inflation"], -.08)

    def test_filing_date_does_not_replace_acceptance_or_public_availability(self):
        question, sources = packet("systematic-001")
        initial = CHECKS["systematic-001"](question, sources)[0]
        sources["systematic-apple-index"] = sources["systematic-apple-index"].replace("2024-02-02", "2024-02-03")
        self.assertEqual(CHECKS["systematic-001"](question, sources)[0], initial)
        sources["systematic-nvidia-index"] = sources["systematic-nvidia-index"].replace("16:36:57", "16:36:58")
        changed = CHECKS["systematic-001"](question, sources)[0]
        self.assertEqual(changed["at_ready_selected_asset"], "apple")
        self.assertEqual(changed["nvidia_public_eligibility_at_acceptance"], "ineligible")

    def test_missing_dissemination_caveat_is_not_silently_certified(self):
        question, sources = packet("systematic-001")
        sources["systematic-sec-time-faq"] = sources["systematic-sec-time-faq"].replace("There is no timestamp to indicate when filing content is first available on sec.gov.", "")
        with self.assertRaisesRegex(ValueError, "limitations absent"):
            CHECKS["systematic-001"](question, sources)


if __name__ == "__main__":
    unittest.main()
