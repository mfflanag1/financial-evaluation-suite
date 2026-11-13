"""Independent accounting identities and branch controls from public excerpts."""
from copy import deepcopy
from datetime import date
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import unittest

from quality import advanced_source_numerical as audit

ROOT = Path(__file__).resolve().parents[1]


def packet(cid):
    folder = ROOT / "questions/legacy/source_grounded/cases" / cid
    question = json.loads((folder / "case.json").read_text())
    return question, {sid: (folder / "sources" / f"{sid}.txt").read_text() for sid in question["source_ids"]}


class AdvancedSourceArithmetic(unittest.TestCase):
    def test_all_declared_fields_are_derived_from_public_packets(self):
        count = 0
        for cid, calculate in audit.CHECKS.items():
            question, sources = packet(cid)
            answers, certificate = calculate(question, sources)
            self.assertEqual(set(answers), {field["id"] for field in question["output_fields"]})
            self.assertIn("retained excerpts", certificate["scope"])
            count += len(answers)
        self.assertEqual(count, 38)

    def test_paypal_consistent_denominator_reverses_reported_eps_comparison(self):
        answers, _ = audit.paypal(*packet("h2-001"))
        comparable = 100 * (F(465, 383) - 1)
        mixed_policy = 100 * (F(465, 510) - 1)
        self.assertAlmostEqual(answers["non_gaap_eps_growth_comparable_pct"], float(comparable))
        self.assertAlmostEqual(answers["old_basis_eps_growth_error_pp"], float(mixed_policy - comparable))
        self.assertGreater(comparable, 0)
        self.assertLess(mixed_policy, 0)
        self.assertEqual(answers["prior_year_non_gaap_income_recast_reduction"], 5646 - 4240)
        self.assertFalse(answers["old_and_new_non_gaap_eps_directly_comparable"])

    def test_paypal_timing_bridge_uses_contribution_sign_and_disjoint_periods(self):
        answers, certificate = audit.paypal(*packet("h2-001"))
        q4 = answers["q42023_positive_bnpl_timing_contribution"]
        nine = answers["first_nine_months_2023_bnpl_timing_contribution"]
        self.assertEqual(q4, 2469 - 774)
        self.assertEqual(nine + q4, 4220 - 4554)
        self.assertEqual(nine, -2029)
        self.assertEqual(F(certificate["annual_timing_contribution"]), -334)
        self.assertAlmostEqual(answers["q42023_timing_share_of_free_cash_flow_pct"], 100 * 1695 / 2469)
        self.assertFalse(answers["sold_bnpl_credit_loss_rate_identified"])

    def test_paypal_source_arithmetic_and_prior_vintage_conflicts_are_rejected(self):
        question, sources = packet("h2-001")
        bad = dict(sources)
        bad["h2a-paypal-2024"] = bad["h2a-paypal-2024"].replace("(133)", "(134)")
        with self.assertRaisesRegex(ValueError, "reconciliation does not balance"):
            audit.paypal(question, bad)
        bad = dict(sources)
        old = bad["h2a-paypal-2023"]
        head, cash = old.split("Reconciliation of Operating Cash Flow", 1)
        bad["h2a-paypal-2023"] = head + "Reconciliation of Operating Cash Flow" + cash.replace("4,220", "4,221")
        with self.assertRaisesRegex(ValueError, "vintages do not reconcile"):
            audit.paypal(question, bad)

    def test_cisco_tax_bridge_reconciles_earnings_and_acquisition_perimeters(self):
        answers, certificate = audit.cisco(*packet("h2-002"))
        self.assertEqual(F(certificate["pretax_decline"]), 1134)
        self.assertEqual(F(certificate["tax_reduction"]), 994)
        self.assertEqual(F(certificate["net_income_change"]), -140)
        self.assertAlmostEqual(answers["pretax_decline_offset_by_tax_reduction_pct"], 100 * 994 / 1134)
        self.assertGreater(answers["reported_revenue_growth_2025_pct"], 0)
        self.assertLess(answers["revenue_change_vs_2024_splunk_proforma_pct"], 0)
        self.assertAlmostEqual(answers["reported_minus_proforma_comparator_growth_pp"],
                               100 * 56654 * (1 / 53803 - 1 / 56761))
        self.assertEqual(answers["fy2024_proforma_minus_reported_net_income"], 9280 - 10320)
        self.assertFalse(answers["comparator_identifies_organic_growth"])
        self.assertFalse(answers["proforma_net_income_is_realized_combined_income"])

    def test_cisco_shareholder_cash_excludes_accounting_awards_and_other_acquisitions(self):
        answers, certificate = audit.cisco(*packet("h2-002"))
        self.assertEqual(answers["shareholder_cash_consideration_net_of_acquired_cash"], 26950 - 2422)
        self.assertEqual(answers["purchase_consideration_exceeds_common_share_cash"], 137 + 3)
        operands = certificate["operands"]
        self.assertEqual(F(operands["purchase_consideration"]),
                         sum(F(operands[key]) for key in ("common_share_cash", "converted_awards", "preexisting_relationships")))

    def test_cisco_inconsistent_income_or_purchase_bridge_is_rejected(self):
        question, sources = packet("h2-002")
        bad = dict(sources)
        bad["h2a-cisco-2025"] = bad["h2a-cisco-2025"].replace("10,180", "10,181")
        with self.assertRaisesRegex(ValueError, "Tax and pretax"):
            audit.cisco(question, bad)
        bad = dict(sources)
        bad["h2a-cisco-2024"] = bad["h2a-cisco-2024"].replace("26,950", "26,951")
        with self.assertRaisesRegex(ValueError, "Purchase-consideration"):
            audit.cisco(question, bad)

    def test_boeing_receipts_modes_and_dividend_stub_are_separate(self):
        answers, certificate = audit.boeing(*packet("h2-005"))
        self.assertEqual(answers["preferred_shares_represented"], 20000 / 20)
        self.assertEqual(answers["first_plus_next_three_dividends"], 20000 * (.625 + 3 * .75))
        self.assertEqual(answers["normal_mandatory_common_share_entitlement"], 20000 * 50 / 160)
        self.assertEqual(answers["ordinary_optional_common_share_entitlement"], 20000 * .2914)
        self.assertEqual(answers["normal_minus_optional_marked_value"], 67520)
        self.assertFalse(answers["single_close_identifies_mandatory_conversion"])
        self.assertFalse(answers["dividends_contractually_guaranteed_in_cash"])
        self.assertIn("fractional settlement", certificate["interpretation"])

    def test_boeing_event_interpolation_uses_182_actual_days_not_half_a_year(self):
        answers, certificate = audit.boeing(*packet("h2-005"))
        # Independently use the depositary-share table, not the preferred table
        # parsed by the audit. No receipt/preferred conversion is needed here.
        early = (F(".3072") + F(".3046")) / 2
        later = (F(".3130") + F(".3090")) / 2
        expected = 20000 * (early + F(182, 365) * (later - early))
        self.assertAlmostEqual(answers["fundamental_change_common_share_entitlement"], float(expected), places=10)
        self.assertEqual(certificate["date_weight"], "182/365")
        self.assertNotEqual(float(expected), float(20000 * (early + later) / 2))
        self.assertAlmostEqual(answers["fundamental_minus_ordinary_optional_shares"], float(expected - 5828), places=10)

    def test_boeing_comparison_mark_and_contractual_market_value_are_distinct_inputs(self):
        question, sources = packet("h2-005")
        baseline, _ = audit.boeing(question, sources)
        changed = deepcopy(question)
        changed["prompt"] = changed["prompt"].replace("with a 160 mark", "with a 170 mark")
        result, _ = audit.boeing(changed, sources)
        self.assertEqual(result["normal_mandatory_common_share_entitlement"], baseline["normal_mandatory_common_share_entitlement"])
        self.assertEqual(result["normal_minus_optional_marked_value"], (6250 - 5828) * 170)

    def test_boeing_exact_table_nodes_and_leap_year_interpolation(self):
        prices = [F(100), F(200)]
        rows = {date(2024, 1, 1): [F(2), F(4)], date(2025, 1, 1): [F(6), F(8)]}
        self.assertEqual(audit._interpolate(F(100), date(2024, 1, 1), prices, rows)[0], 2)
        self.assertEqual(audit._interpolate(F(200), date(2025, 1, 1), prices, rows)[0], 8)
        value, certificate = audit._interpolate(F(150), date(2024, 7, 1), prices, rows)
        self.assertEqual(value, 3 + F(182, 366) * 4)
        self.assertEqual(certificate["days_in_interval"], 366)
        with self.assertRaisesRegex(ValueError, "within the retained table"):
            audit._interpolate(F(99), date(2024, 7, 1), prices, rows)

    def test_ibm_printed_percentages_conserve_basis_and_keep_blocks_separate(self):
        answers, _ = audit.ibm(*packet("h2-007"))
        total_basis = 115 * 80 + 52 * 140
        self.assertEqual(answers["retained_ibm_basis"], total_basis * .958)
        disposed = F(52 * 140) * F(42, 1000) * F(2, 5) / F(52, 5)
        self.assertEqual(answers["fractional_kd_basis"], float(disposed))
        self.assertAlmostEqual(sum(answers[key] for key in ("retained_ibm_basis", "retained_kd_basis", "fractional_kd_basis")), total_basis)
        self.assertNotEqual(float(disposed), float(F(total_basis) * F(42, 1000) * F(2, 167)))
        self.assertEqual(answers["retained_ibm_shares"], 167)
        self.assertEqual(answers["kyndryl_economic_entitlement"], 33.4)

    def test_ibm_unrelated_block_basis_and_sale_cash_have_separate_effects(self):
        question, sources = packet("h2-007")
        baseline, _ = audit.ibm(question, sources)
        changed = deepcopy(question)
        changed["prompt"] = changed["prompt"].replace("$80 each", "$800 each")
        result, _ = audit.ibm(changed, sources)
        self.assertEqual(result["fractional_kd_basis"], baseline["fractional_kd_basis"])
        self.assertGreater(result["retained_ibm_basis"], baseline["retained_ibm_basis"])
        changed = deepcopy(question)
        changed["prompt"] = changed["prompt"].replace("charging $0.50", "charging $5.50")
        result, _ = audit.ibm(changed, sources)
        self.assertEqual(result["fractional_kd_basis"], baseline["fractional_kd_basis"])
        self.assertEqual(result["cash_in_lieu_net"], baseline["cash_in_lieu_net"] - 5)
        self.assertAlmostEqual(result["fractional_disposal_gain"], baseline["fractional_disposal_gain"] - 5)
        self.assertFalse(baseline["actual_cash_known_without_sale_price"])
        self.assertFalse(baseline["issuer_method_is_mandatory"])

    def test_ibm_printed_allocation_and_fraction_ownership_are_validated(self):
        question, sources = packet("h2-007")
        bad = dict(sources)
        bad["h2b-ibm-tax"] = bad["h2b-ibm-tax"].replace("95.8%", "95.7%")
        with self.assertRaisesRegex(ValueError, "sum to one"):
            audit.ibm(question, bad)
        changed = deepcopy(question)
        changed["prompt"] = changed["prompt"].replace("115 IBM shares", "116 IBM shares")
        with self.assertRaisesRegex(ValueError, "fraction ownership"):
            audit.ibm(changed, sources)

    def test_public_derivations_import_no_scorer_or_answer_modules(self):
        script = """
import json,sys
from pathlib import Path
from quality.advanced_source_numerical import CHECKS
for cid,fun in CHECKS.items():
 p=Path('questions/legacy/source_grounded/cases')/cid
 q=json.loads((p/'case.json').read_text())
 fun(q,{s:(p/'sources'/f'{s}.txt').read_text() for s in q['source_ids']})
assert not any(n.startswith(('scoring_v2','scoring_v3','private_evaluator','financial_evals')) for n in sys.modules)
"""
        subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
