"""Public-input arithmetic, cash boundaries and non-identification controls."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction as F
import json
import math
from pathlib import Path
import unittest

from quality import source_remaining as audit


ROOT = Path(__file__).resolve().parents[1]


def packet(case_id):
    folder = ROOT / "questions/legacy/source_grounded/cases" / case_id
    question = json.loads((folder / "case.json").read_text())
    return question, {sid: (folder / "sources" / f"{sid}.txt").read_text() for sid in question["source_ids"]}


def calculate(case_id):
    return audit.CHECKS[case_id](*packet(case_id))


class SourceScopeTests(unittest.TestCase):
    def test_all_25_packets_derive_all_256_declared_outputs(self):
        count = 0
        self.assertEqual(len(audit.CHECKS), 25)
        for cid, check in audit.CHECKS.items():
            question, sources = packet(cid)
            answers, certificate = check(question, sources)
            self.assertEqual(set(answers), {field["id"] for field in question["output_fields"]})
            self.assertEqual(certificate["canonical_public_packet_sha256"], audit.packet_hash(question, sources))
            self.assertIn("assistant arithmetic review", certificate["scope"])
            self.assertEqual(set(certificate["exact_results"]), set(answers))
            for field in question["output_fields"]:
                value = answers[field["id"]]
                if field["type"] == "boolean":
                    self.assertIs(type(value), bool, (cid, field["id"]))
                elif field["type"] == "number":
                    self.assertIsNot(type(value), bool)
                    self.assertTrue(math.isfinite(value))
                    self.assertAlmostEqual(float(F(str(certificate["exact_results"][field["id"]]))), value)
                else:
                    self.assertIsInstance(value, str)
            count += len(answers)
        self.assertEqual(count, 256)

    def test_changed_question_or_retained_text_requires_new_review(self):
        for cid, check in audit.CHECKS.items():
            question, sources = packet(cid)
            changed = deepcopy(question)
            changed["prompt"] += " Changed cash assumption."
            with self.assertRaisesRegex(ValueError, "differs from reviewed"):
                check(changed, sources)
            altered = dict(sources)
            altered[next(iter(altered))] += "\nChanged source version."
            with self.assertRaisesRegex(ValueError, "differs from reviewed"):
                check(question, altered)
            altered = dict(sources, unreviewed="Extra material")
            with self.assertRaises(ValueError):
                check(question, altered)


class CorporateActionTests(unittest.TestCase):
    def test_natwest_final_cash_controls_basket_and_break_even(self):
        answers, _ = calculate("vol-001")
        self.assertEqual(answers["book_expiry_pnl"], 20 * (92 * F("5.2") + F("42.3") - 500 - 17))
        self.assertNotEqual(F("42.3"), 100 * F(".381441") + F("4.16"))
        break_even = (F(500 + 17) - F("42.3")) / 92
        r = audit.adjusted_call(F(92), F("42.3"), break_even, F(5), F(".17"), 20)
        self.assertEqual(r["pnl"], 0)
        below = audit.adjusted_call(F(92), F("42.3"), F(1), F(5), F(".17"), 20)
        self.assertEqual(below["intrinsic"], 0)
        self.assertEqual(below["pnl"], -340)

    def test_ge_fixed_cash_has_zero_spot_slope_and_unchanged_strike_multiplier(self):
        base = audit.adjusted_call(F(12), F("52.22"), F(120), F(14), F(".25"), 50)
        higher = audit.adjusted_call(F(12), F("52.22"), F(121), F(14), F(".25"), 50)
        self.assertEqual(higher["pnl"] - base["pnl"], 600)
        answers, _ = calculate("vol-002")
        self.assertAlmostEqual(answers["final_equivalent_quote"], float((12 * F(120) + F("52.22")) / 100))
        self.assertEqual(answers["book_strike_payment"], 50 * 1400)
        self.assertEqual(answers["book_expiry_pnl"], float(base["pnl"]))

    def test_mmm_two_asset_stress_and_premium_break_even(self):
        answers, _ = calculate("vol-003")
        old = 40 * max(100 * 92 + 25 * 80 - 11000, 0)
        new = 40 * max(100 * 93 + 25 * 77 - 11000, 0)
        self.assertEqual(new - old, answers["stress_pnl_change"])
        self.assertEqual(100 * 92 + 25 * F(str(answers["solv_breakeven"])), 11000 + 120)
        self.assertFalse(answers["settlement_allocation_proves_tax_basis"])

    def test_collar_separates_average_from_mark_and_bounds_all_branches(self):
        answers, _ = calculate("event-001")
        self.assertEqual(audit.att_ratio(F(40)), F("1.344"))
        for average in (F(1), F("37.4109"), F("37.411"), F(40), F("41.349"), F(1000)):
            ratio = audit.att_ratio(average)
            self.assertGreaterEqual(ratio, F("1.3"))
            self.assertLessEqual(ratio, F("1.437"))
        # A remains 40; a higher separate settlement mark changes value, not ratio.
        self.assertEqual(F("53.75") + audit.att_ratio(F(40)) * 42, F("110.198"))
        self.assertEqual(answers["b_short_hedge_shares"], 1344)
        with self.assertRaises(ValueError):
            audit.att_ratio(F(0))

    def test_election_allocation_conserves_exact_cash_quota_in_every_small_partition(self):
        minimum_stock = F(1)
        maximum_stock = F(0)
        for cash in range(21):
            for stock in range(21 - cash):
                none = 20 - cash - stock
                allocation = audit.election_fractions(20, cash, stock, none)
                self.assertEqual(cash * allocation["cash"] + stock * allocation["stock"] + none * allocation["none"], 10)
                self.assertTrue(all(0 <= part <= 1 for part in allocation.values()))
                if stock >= 2:
                    fraction = 1 - allocation["stock"]
                    minimum_stock = min(minimum_stock, fraction)
                    maximum_stock = max(maximum_stock, fraction)
        self.assertEqual((minimum_stock, maximum_stock), (F(1, 2), F(1)))
        self.assertEqual(audit.election_fractions(20, 10, 10, 0)["stock"], 0)
        with self.assertRaises(ValueError):
            audit.election_fractions(20, 10, 10, 1)

    def test_vmware_uses_historical_exchange_ratio_and_exact_profit_policy(self):
        answers, _ = calculate("event-002")
        self.assertEqual(answers["s_holder_cash"] / 142.5 + answers["s_holder_broadcom_shares"] / .252, 800)
        self.assertEqual(answers["s_holder_net_profit"], 42750 + 126 * 600 - 800 * F("146.5"))
        self.assertTrue(answers["s_meets_profit_policy"])
        self.assertEqual(answers["u_marked_entitlement_upper_bound"] - answers["u_marked_entitlement_lower_bound"], 4350)

    def test_ticking_schedule_resets_at_each_tier_and_has_no_may22_value(self):
        self.assertEqual(audit.ticking_fee(date(2023, 2, 22)), F(".15333333"))
        self.assertEqual(audit.ticking_fee(date(2023, 3, 22)), F(".22333333"))
        self.assertEqual(audit.ticking_fee(date(2023, 4, 22)), F(".3266667"))
        self.assertEqual(audit.ticking_fee(date(2023, 5, 21)), F(".3266667") + 29 * F(".00416667"))
        with self.assertRaises(ValueError):
            audit.ticking_fee(date(2023, 5, 22))
        answers, certificate = calculate("event-003")
        probability = F(certificate["exact_results"]["may21_break_even_success_probability"])
        financing = 23 * F(".08") * F(88, 365)
        success = 24 + audit.ticking_fee(date(2023, 5, 21)) - 23 - F(".05") - financing
        failure = 17 - 23 - F(".05") - financing
        self.assertEqual(probability * success + (1 - probability) * failure, 0)
        self.assertLess(answers["april21_to_may21_incremental_net_profit"], 0)

    def test_odd_lot_acceptance_uses_counts_and_preserves_fractional_economic_interest(self):
        answers, certificate = calculate("h2-008")
        fraction = F(certificate["exact_results"]["nonodd_acceptance_fraction"])
        self.assertEqual(fraction * 793920159 + 6529526, 190955436)
        self.assertEqual(answers["oddlot_whole_kvue_shares"] + F(str(answers["oddlot_fractional_interest"])), 99 * F("8.0324"))
        self.assertEqual(F(str(answers["oddlot_marked_economic_profit"])), 99 * (F("8.0324") * 23 - 167))
        self.assertFalse(answers["actual_fractional_cash_known"])


class CreditAccountingTests(unittest.TestCase):
    def test_arcosa_cash_cap_and_inclusive_ebitda_boundary(self):
        base = audit.leverage_metrics(F(2200), F(200), F(150), F(420), F(5))
        extra_cash = audit.leverage_metrics(F(2200), F(1000), F(150), F(420), F(5))
        self.assertEqual(base, extra_cash)
        boundary = audit.leverage_metrics(F(2200), F(100), F(150), F(420), F(5))
        self.assertEqual(boundary["leverage"], 5)
        self.assertEqual(audit.leverage_metrics(F(2200), F(200), F(150), F(410), F(5))["leverage"], 5)
        self.assertGreater(audit.leverage_metrics(F(2200), F(0), F(150), F(420), F(5))["leverage"], 5)
        answers, _ = calculate("credit-001")
        self.assertEqual(answers["covenant_ebitda"], 180 + sum((60, 45, 95, 25, 20, 15, 25)) - sum((5, 10, 30)))
        self.assertEqual(answers["september_2025_external_paydown_needed"], 2200 - 150 - F("4.5") * 420)

    def test_vestis_debt_and_cash_perimeters_do_not_net_other_groups(self):
        answers, certificate = calculate("credit-002")
        self.assertEqual(answers["included_indebtedness"], 1800 + 100 + 20 + 10)
        self.assertEqual(answers["eligible_cash"], 130 + 40 + 30)
        self.assertEqual(answers["excluded_cash"], 25 + 90)
        minimum = F(certificate["exact_results"]["minimum_ebitda_at_boundary"])
        self.assertEqual(minimum * F("4.5"), answers["net_debt"])
        self.assertLess(380, minimum)
        self.assertFalse(answers["interest_coverage_compliance_identifiable"])

    def test_springing_trigger_is_strict_but_active_ratio_allows_equality(self):
        boundary = F(850) * F(".35")
        self.assertEqual(audit.springing_compliance(boundary, 850, 999999, 1), (False, True))
        self.assertEqual(audit.springing_compliance(boundary + F(1, 1000), 850, 2025, 300), (True, True))
        self.assertEqual(audit.springing_compliance(boundary + F(1, 1000), 850, F("2025.001"), 300), (True, False))
        answers, _ = calculate("credit-003")
        self.assertEqual(290 - answers["stress_minimum_revolver_repayment_to_deactivate"] + F("27.5"), boundary)
        self.assertEqual((2020 - answers["stress_term_debt_repayment_to_meet_active_ratio"]) / 280, 6.75)

    def test_carnival_liquidity_and_coverage_joint_feasibility_boundary(self):
        required = audit.coverage_repayment(F(3000), F(1200), F(3000), F(1500), F(3), F(".1"))
        self.assertEqual(required["required"], 2000)
        self.assertEqual(required["maximum"], 1500)
        self.assertFalse(required["feasible"])
        equality = audit.coverage_repayment(F(3000), F(1200), F(3500), F(1500), F(3), F(".1"))
        self.assertTrue(equality["feasible"])
        self.assertEqual(equality["coverage"], 3)
        self.assertEqual(equality["ebitda_gap"], 0)

    def test_accounting_reconciliations_preserve_perimeters_and_cash_vs_accrual(self):
        altice, _ = calculate("equity-001")
        self.assertEqual(altice["metric_gap"], altice["adjusted_ebitda"] - altice["operating_cash_flow"])
        self.assertEqual(altice["free_cash_flow"] + altice["minimum_capex_reduction"], 150000)
        self.assertFalse(altice["replacement_computable"])
        intel, _ = calculate("equity-002")
        self.assertEqual(intel["recast_2023_consolidated_revenue"], 29258 + 12635 + 5774 + 18910 + 5608 - 17957)
        self.assertNotEqual(intel["dcai_2023_basis_difference"], 12817 - 12635)
        msft, _ = calculate("equity-003")
        self.assertEqual(-1362 - 400 + F(str(msft["hypothetical_cost_synergy_required"])), F(".05") * 5729)
        self.assertFalse(msft["residual_growth_is_identified_organic_growth"])

    def test_danaher_total_financing_removes_both_transfers_with_correct_sign(self):
        answers, _ = calculate("h2-003")
        self.assertEqual(answers["continuing_free_cash_flow_2023"], 6490 - 1383 + 12)
        self.assertEqual(answers["financing_cash_flow_without_separation_transfers"] + 2600 - 427, 154 - 427)
        self.assertEqual(answers["fy2022_operating_cash_removed_from_continuing"], 906)
        self.assertFalse(answers["veralto_debt_remains_danaher_obligation_after_separation"])

    def test_preferred_redemption_unit_scale_and_premium_only_payback(self):
        answers, _ = calculate("h2-006")
        self.assertEqual(F(str(answers["preferred_face_redeemed_2023"])) + F(str(answers["preferred_face_remaining_end2023"])), 10000)
        self.assertEqual(answers["cash_dividend_reduction_2024_vs2023"], 83)
        self.assertNotEqual(answers["cash_dividend_reduction_2024_vs2023"], answers["common_income_charge_reduction_2024_vs2023"])
        self.assertEqual(answers["premium_only_recovery_years"], .1 / .08)
        self.assertEqual(answers["synthetic_redemption_cash_outlay"], 110)


class EnergyTreasuryTests(unittest.TestCase):
    def test_generator_financial_swap_persists_during_reduced_availability(self):
        args = [F(1054), 120, F(".8"), F("7.2"), F(65), F(3), F(".5"), F(3), 600, F(60), F(68)]
        base = audit.generator_margin(*args)
        reduced = list(args)
        reduced[2] /= 2
        changed = audit.generator_margin(*reduced)
        self.assertEqual(changed["generation"] * 2, base["generation"])
        self.assertEqual(changed["hedge"], base["hedge"])
        shocked = list(args)
        shocked[4] += 1
        shocked[10] += 1
        self.assertEqual(audit.generator_margin(*shocked)["total"] - base["total"], base["hub_sensitivity"])
        shocked = list(args)
        shocked[6] += 1
        self.assertEqual(audit.generator_margin(*shocked)["total"] - base["total"], base["basis_sensitivity"])
        shocked[6] = base["break_even_basis"]
        self.assertEqual(audit.generator_margin(*shocked)["total"], 0)

    def test_basis_hedge_cancels_monthly_basis_but_not_outright_or_shape(self):
        # Physical price decomposes as HH 2.2 + monthly basis -2.5 + shape .2.
        args = [F(62500), 2500, F("-.1"), F("-.6"), F("-.3"), F("2.2"), 4]
        base = audit.basis_hedge(*args)[3]
        basis_shock = list(args)
        basis_shock[2] += 1
        basis_shock[4] += 1
        self.assertEqual(audit.basis_hedge(*basis_shock)[3], base)
        hh_shock = list(args)
        for i in (2, 4, 5):
            hh_shock[i] += 1
        self.assertEqual(audit.basis_hedge(*hh_shock)[3] - base, 62500)
        shape = list(args)
        shape[2] += 1
        self.assertEqual(audit.basis_hedge(*shape)[3] - base, 62500)
        invalid = list(args)
        invalid[0] += 1
        with self.assertRaises(ValueError):
            audit.basis_hedge(*invalid)

    def test_pjm_vintage_change_holds_price_and_delivery_days_fixed(self):
        answers, _ = calculate("h2-010")
        delta = -365 * (F("444.26") + F("466.35") + 2 * F("269.92"))
        self.assertEqual(F(str(answers["future_rating_vintage_revenue_change"])), delta)
        uplift = 365 * (79 * (F("444.26") - F("269.92")) + 14 * (F("466.35") - F("269.92")))
        self.assertEqual(F(str(answers["locational_revenue_uplift"])), uplift)
        self.assertAlmostEqual(answers["storage_break_even_capacity_price"] * 59 * 365, 6000000)

    def test_svb_bridge_principal_is_not_free_cash_and_accounts_aggregate(self):
        insured, bridge, shortfall, surplus = audit.treasury_bridge(F(1000000), F(250000), F(200000), F(250000), F(750000), F(100000), F(".01"))
        self.assertEqual((insured, bridge, shortfall, surplus), (250000, 50000, 650500, 99500))
        self.assertEqual(1000000 + 200000 - 250000 - 750000 - 100000 - 500, surplus)
        no_bridge = audit.treasury_bridge(F(1000000), F(250000), F(250000), F(250000), F(750000), F(100000), F(".01"))
        self.assertEqual(no_bridge[1], 0)
        self.assertEqual(no_bridge[3], 150000)

    def test_at1_lookthrough_and_tier2_mark_do_not_become_available_equity_cash(self):
        answers, _ = calculate("h2-012")
        self.assertEqual(answers["shared_at1_mark_loss"], 800000 + 1050000)
        self.assertEqual(answers["aggregate_portfolio_mark_pnl"], -1850000 - 26960 + 2000 - 30000)
        self.assertEqual(answers["immediate_cash_financing_need"], 925000 - 120000 - 2000)
        self.assertEqual(answers["eventual_net_ubs_share_exposure"], 0)
        self.assertFalse(answers["tier2_zero_required_by_write_down"])


class PortfolioReplayTests(unittest.TestCase):
    def test_separation_ledger_field_uses_millions_but_participation_uses_dollars(self):
        question, _ = packet("portfolio-001")
        units = {field["id"]: field["unit"] for field in question["output_fields"]}
        self.assertEqual(units["financing_ledger_difference"], "USD_million")
        self.assertEqual(units["credit_participation_principal"], "USD")
        answers, _ = calculate("portfolio-001")
        self.assertEqual(answers["financing_ledger_difference"], 664 - 663)
        self.assertEqual(answers["credit_participation_principal"], 500 * 1000000 * F(".0002"))

    def test_all_plan_feasibility_boundaries_are_inclusive_for_both_books(self):
        for rows in (audit.separation_book()[2], audit.freeze_book()[3]):
            self.assertEqual(set(rows), set(audit.PLAN_ORDER))
            for code, row in rows.items():
                self.assertIn(code, audit.select_plan(rows, row["required_cash"])["feasible"])
                self.assertNotIn(code, audit.select_plan(rows, row["required_cash"] - F(1, 1000000))["feasible"])
                self.assertEqual(row["required_cash"], row["t0_locked"] + max(row["t1_outflow"]))

    def test_optimizer_changes_only_at_exact_feasibility_thresholds(self):
        for rows, budget, target, threshold in ((audit.separation_book()[2], 18000, "AC", 20850),
                                                (audit.freeze_book()[3], 23200, "AB", 24300)):
            self.assertEqual(audit.select_plan(rows, budget)["optima"], [target])
            self.assertEqual(audit.funding_result(rows, budget)["first_optimizer_change_budget"], threshold)
            self.assertEqual(audit.select_plan(rows, threshold - F(1, 1000000))["optima"], [target])
            self.assertEqual(audit.select_plan(rows, threshold)["optima"], ["ABC"])

    def test_favorable_hedge_cannot_finance_gross_negative_hedge_call(self):
        rows = audit.funding_plans([0], [0], {"A": [-100], "B": [100], "C": [0]},
                                   {"A": 0, "B": 0, "C": 0}, {"A": 0, "B": 0, "C": 0})
        self.assertEqual(rows["AB"]["terminal_pnl"], [0])
        self.assertEqual(rows["AB"]["required_cash"], 100)
        self.assertNotIn("AB", audit.select_plan(rows, 99)["feasible"])

    def test_refundable_margin_changes_cash_not_terminal_profit(self):
        original = audit.separation_book(F(1000))[2]
        changed = audit.separation_book(F(5000))[2]
        for code in audit.PLAN_ORDER:
            self.assertEqual(original[code]["terminal_pnl"], changed[code]["terminal_pnl"])
            self.assertEqual(changed[code]["required_cash"] - original[code]["required_cash"], 4000 if "C" in code else 0)

    def test_missing_margin_and_dispatch_have_distinct_feasible_optimizer_witnesses(self):
        margin_optima = [audit.select_plan(audit.separation_book(F(m))[2], 18000)["optima"] for m in (0, 5000)]
        dispatch_optima = [audit.select_plan(audit.freeze_book(F(d))[3], 23200)["optima"] for d in (40, 100)]
        self.assertTrue(all(margin_optima) and all(dispatch_optima))
        self.assertNotEqual(*margin_optima)
        self.assertNotEqual(*dispatch_optima)

    def test_freeze_fuel_and_replacement_power_use_different_volumes(self):
        _, base, early, _ = audit.freeze_book()
        producer = 500 * F("33.86") - 1000 * F("3.86")
        generator = (10000 - 40 * 7 * F("33.86") - 60 * 250) - (10000 - 100 * 7 * F("3.86"))
        self.assertEqual(base[0], producer + generator - 300 * 20)
        self.assertEqual(early[0], 60 * 250 + 300 * 20)
        self.assertEqual(audit.freeze_book(F(100))[2][0], 6000)

    def test_vintage_readiness_is_inclusive_and_ties_go_to_peer(self):
        ready = datetime(2024, 2, 14, 21, 45, 30, tzinfo=timezone.utc)
        select = lambda time: audit.vintage_choice(time, ready, F("6.6"), F("2.1"), F(3))
        self.assertEqual(select(ready - timedelta(microseconds=1)), "lyft")
        self.assertEqual(select(ready), "peer")
        self.assertEqual(audit.vintage_choice(ready, ready, F(9), F(3), F(3)), "peer")

    def test_fixed_sizing_costs_use_changed_exit_notional_and_do_not_compound(self):
        returns = [{"lyft": F("-.08"), "peer": F(".01")}, {"lyft": F(".02"), "peer": F(".01")}]
        profit, ledger = audit.fixed_round_replay(["lyft", "peer"], returns)
        self.assertEqual(profit, -800 + 100 - 10 - F("9.2") - 10 - F("10.1"))
        self.assertEqual([row["entry_notional"] for row in ledger], [10000, 10000])
        self.assertEqual([row["exit_notional"] for row in ledger], [9200, 10100])
        backfilled, _ = audit.fixed_round_replay(["peer", "peer"], returns)
        self.assertEqual(backfilled - profit, F("899.1"))
        with self.assertRaises(ValueError):
            audit.fixed_round_replay(["lyft"], returns[:1], cash=F(10000))

    def test_identity_repair_preserves_exit_losses_dividend_cash_and_sale_fee(self):
        answers, _ = calculate("systematic-003")
        self.assertEqual(F(str(answers["valid_terminal_equity"])), 60 * 102 + 60 + 80 * 20 - F("6.12"))
        break_even = F(str(answers["exit_proceeds_break_even"]))
        self.assertEqual(60 * 102 + 60 + 80 * break_even - F("6.12"), 10000)
        self.assertEqual(answers["rename_required_turnover"], 0)
        self.assertFalse(answers["ticker_alone_unique_identity"])


if __name__ == "__main__":
    unittest.main()
