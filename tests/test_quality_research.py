"""Independent financial controls over every authored public research stage."""
import copy
import math
import unittest

from quality.public_contracts import load_research_packet
from quality.research_numerical import audit_case


def result(number):
    return audit_case(load_research_packet(f"q{number}"))


def answers(number, turn=1):
    return result(number)["answers_by_turn"][str(turn)]


class ResearchNumericalTests(unittest.TestCase):
    def test_every_authored_field_and_turn_has_a_finite_derivation(self):
        cases = turns = fields = 0
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, list):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        for n in range(107, 165):
            packet = load_research_packet(f"q{n}")
            calculated = audit_case(packet)
            with self.subTest(case=n):
                self.assertEqual(calculated["unsupported"], [])
                self.assertEqual(calculated["ambiguities"], [])
                self.assertTrue(finite(calculated))
                self.assertEqual(set(calculated["answers_by_turn"]), {str(t["turn"]) for t in packet["turns"]})
                for turn in packet["turns"]:
                    self.assertEqual(set(calculated["answers_by_turn"][str(turn["turn"])]), {o["name"] for o in turn["outputs"]})
                    turns += 1
                    fields += len(turn["outputs"])
            cases += 1
        self.assertEqual((cases, turns), (58, 142))
        self.assertGreater(fields, 700)

    def test_no_later_packet_changes_an_earlier_answer(self):
        for n in range(137, 165):
            packet = load_research_packet(f"q{n}")
            complete = audit_case(packet)["answers_by_turn"]
            for length in range(1, len(packet["turns"])):
                prefix = copy.deepcopy(packet)
                prefix["turns"] = prefix["turns"][:length]
                with self.subTest(case=n, prefix=length):
                    self.assertEqual(audit_case(prefix)["answers_by_turn"], {str(t): complete[str(t)] for t in range(1, length + 1)})

    def test_packet_is_not_mutated_by_calculations(self):
        for n in range(107, 165):
            packet = load_research_packet(f"q{n}")
            before = copy.deepcopy(packet)
            audit_case(packet)
            with self.subTest(case=n):
                self.assertEqual(packet, before)

    def test_withdrawn_and_unavailable_estimates_cannot_enter_consensus(self):
        packet = load_research_packet("q108")
        baseline = audit_case(packet)
        for row in packet["turns"][0]["inputs"]["rows"]:
            if row[0] in ("c2", "a4", "a5"):
                row[5] = 999999
        self.assertEqual(audit_case(packet), baseline)
        first = baseline["answers_by_turn"]["1"]
        self.assertFalse(first["every_matched_broker_raised_eps"])
        self.assertGreater(first["headline_revision_pct"], first["matched_cohort_revision_pct"])

    def test_attribution_does_not_create_causal_identification(self):
        event = answers(111)
        self.assertGreater(event["matched_difference"], 0)
        self.assertEqual(event["earnings_contribution_bounds"][0], 0)
        self.assertFalse(event["positive_earnings_effect_identified"])
        credit = answers(125)
        self.assertAlmostEqual(sum(credit["base_currency_attribution"]), credit["net_base_pnl"])
        self.assertFalse(credit["news_causality_identified"])
        self.assertEqual(result(125)["certificate"]["1"]["cash_wealth_reconciliation"], 0)

    def test_robust_orders_need_both_revenue_and_cash(self):
        packet = load_research_packet("q114")
        baseline = audit_case(packet)["answers_by_turn"]["1"]
        self.assertGreaterEqual(baseline["guaranteed_post_order_revenue"], packet["turns"][0]["inputs"]["revenue_floor"])
        self.assertLessEqual(baseline["incremental_production_cash"], baseline["incremental_receipts"] + 1)
        packet["turns"][0]["inputs"]["opening_unrestricted_cash"] = 100
        relaxed = audit_case(packet)["answers_by_turn"]["1"]
        self.assertGreater(relaxed["funded_qualifying_plan_count"], baseline["funded_qualifying_plan_count"])
        self.assertLess(relaxed["incremental_production_cash"], baseline["incremental_production_cash"])

    def test_partial_identification_preserves_the_base_cost_equation(self):
        record = answers(116)
        self.assertEqual(record["endpoint_base_costs"], [160, 160])
        self.assertLess(record["next_cash_bounds"][0], 80)
        self.assertGreater(record["next_cash_bounds"][1], 80)
        self.assertEqual(record["hurdle_identification"], "not_identified")
        packet = load_research_packet("q116")
        baseline = audit_case(packet)
        packet["turns"][0]["inputs"]["earlier_mixed_segment_quarter"]["cost"] = 99999
        self.assertEqual(audit_case(packet), baseline)

    def test_curve_par_prices_reconcile_and_later_quote_is_unavailable(self):
        packet = load_research_packet("q119")
        baseline = audit_case(packet)
        self.assertEqual(baseline["certificate"]["1"]["par_pricing_residuals"], [0, 0, 0])
        for row in packet["turns"][0]["inputs"]["quotes"]:
            if row["published"] > packet["turns"][0]["inputs"]["cutoff"] or row["currency"] != "USD":
                row["par_rate"] = 99
        self.assertEqual(audit_case(packet), baseline)

    def test_storage_policy_is_nonanticipative_and_pathwise_funded(self):
        packet = load_research_packet("q123")
        i = packet["turns"][0]["inputs"]
        record = audit_case(packet)["answers_by_turn"]["1"]
        q0, low, high = record["nonanticipative_purchase_policy"]
        self.assertEqual(record["leaf_purchase_cash"], [q0*i["initial_price"] + q*i["signal_prices"][s] for s, q in ((0, low), (0, low), (1, high), (1, high))])
        self.assertLessEqual(max(record["leaf_purchase_cash"]), i["purchase_cash_budget"])
        self.assertLessEqual(max(record["leaf_shortfalls"]), i["maximum_leaf_shortfall"])
        self.assertGreaterEqual(min(record["leaf_ending_inventory"]), i["terminal_reserve"])
        self.assertGreaterEqual(record["value_of_early_demand_information"], 0)
        self.assertAlmostEqual(record["expected_total_cost"] - record["early_demand_information_cost"], record["value_of_early_demand_information"])

    def test_redemption_loss_and_fees_conserve_total_investor_value(self):
        record = answers(129)
        self.assertEqual(result(129)["certificate"]["1"]["nav_identity_residual"], 0)
        self.assertGreaterEqual(record["ending_cash"], 1000)
        self.assertLess(record["maximum_common_gate"], 1)
        self.assertFalse(record["full_requested_redemptions_fundable"])
        self.assertAlmostEqual(record["remaining_investor_nav"] + record["redemption_cash_paid"] + record["fees_paid_cash"] + record["realized_sale_loss"], 180 * 120)

    def test_confirmatory_pvalues_cannot_select_siblings(self):
        packet = load_research_packet("q132")
        baseline = audit_case(packet)
        for row in packet["turns"][0]["inputs"]["trials"]:
            if row[0] == "A2":
                row[4] = 0
                row[5] = 999999
        self.assertEqual(audit_case(packet), baseline)

    def test_censored_months_use_joint_completions_and_full_denominator(self):
        record = answers(133)
        self.assertGreater(record["complete_case_gross_mean"], max(record["candidate_net_mean_bounds"]))
        self.assertEqual(record["candidate_net_mean_bounds"], [min(record["net_mean_by_completion"]), max(record["net_mean_by_completion"])])
        self.assertEqual(record["robust_candidate_weight"], 0)

    def test_reclassification_does_not_receive_historical_cash_twice(self):
        stages = result(137)["answers_by_turn"]
        self.assertNotEqual(stages["1"]["normalized_operating_fcf"], stages["2"]["normalized_operating_fcf"])
        self.assertEqual(stages["1"]["usable_cash"], stages["2"]["usable_cash"])
        self.assertEqual(stages["1"]["funding_gap"], stages["2"]["funding_gap"])
        self.assertEqual(stages["3"]["normalized_operating_fcf"], stages["4"]["normalized_operating_fcf"])

    def test_historical_retention_is_not_rewritten_by_future_nonrenewal(self):
        stages = result(141)["answers_by_turn"]
        self.assertEqual(stages["3"]["existing_customer_nrr_pct"], stages["4"]["historical_nrr_pct"])
        self.assertLess(stages["4"]["next_renewal_original_cohort_pct"], stages["4"]["historical_nrr_pct"])

    def test_intramonth_reserve_changes_which_action_is_best(self):
        stages = result(144)["answers_by_turn"]
        self.assertEqual(stages["3"]["chosen_action"], "asset_sale")
        self.assertEqual(stages["4"]["chosen_action"], "layoffs")
        self.assertGreaterEqual(stages["4"]["minimum_pre_receipt_cash"], 10)

    def test_corrected_bayes_model_restarts_from_prior_and_does_not_count_duplicate(self):
        packet = load_research_packet("q145")
        record = audit_case(packet)["answers_by_turn"]
        corrected = [0.5*0.8, 0.3*0.5, 0.2*0.4]
        for actual, joint in zip(record["2"]["state_probabilities"], corrected):
            self.assertAlmostEqual(actual, joint / sum(corrected))
        self.assertEqual(record["2"]["evidence_probability"], sum(corrected))
        packet["turns"][1]["inputs"]["second_positive_notice"] *= 10
        self.assertEqual(audit_case(packet)["answers_by_turn"], record)
        self.assertFalse(record["3"]["joint_policy_met"])

    def test_derivative_cash_timing_changes_liquidity_not_investment_pnl(self):
        stages = result(150)["answers_by_turn"]
        self.assertEqual(stages["2"]["combined_investment_pnl"], stages["3"]["combined_investment_pnl"])
        self.assertEqual(stages["3"]["combined_investment_pnl"], stages["4"]["combined_investment_pnl"])
        self.assertAlmostEqual(stages["3"]["cash_shortfall"] - stages["4"]["cash_shortfall"], stages["4"]["protection_receipt"])

    def test_option_payoff_certificate_holds_between_knots_and_in_tail(self):
        for turn in range(1, 5):
            record = answers(154, turn)
            cash, *calls = record["selected_cash_and_calls"]
            for spot in [0, 30, 89.9, 90, 95, 100, 105, 110, 115, 10000]:
                self.assertGreaterEqual(cash + sum(q*max(spot-k, 0) for q, k in zip(calls, (90, 100, 110))), 0)
            self.assertGreaterEqual(sum(calls), 0)
        self.assertFalse(answers(154, 3)["negative_cost_nonnegative_payoff_exists"])
        self.assertTrue(answers(154, 4)["negative_cost_nonnegative_payoff_exists"])

    def test_fair_gate_and_bridge_interest_are_separate(self):
        stages = result(157)["answers_by_turn"]
        for turn in (1, 2, 3, 4):
            record = stages[str(turn)]
            self.assertAlmostEqual(record["redemption_payments"][0] / record["redemption_payments"][1], 18 / 12)
            self.assertLessEqual(sum(record["redemption_payments"]) / record["nav_per_unit"], 20)
        final = stages["4"]
        self.assertAlmostEqual(final["asset_sale"], final["bridge_draw"]*1.01)
        self.assertAlmostEqual(final["ending_nav"], 90 + 4 - sum(final["redemption_payments"]) - final["bridge_draw"]*0.01)

    def test_future_netting_does_not_fund_current_margin(self):
        final = answers(158, 4)
        self.assertGreater(final["current_margin"], final["margin_after_approval"])
        self.assertLessEqual(abs(final["remaining_delta"]), 2)
        self.assertGreaterEqual(final["free_cash_now"], 0)
        self.assertAlmostEqual(final["free_cash_now"] + final["current_margin"] + sum(final["contracts_closed"])*50, 12000)

    def test_futures_cash_reconciliation_does_not_pay_notional(self):
        final = answers(159, 4)
        self.assertEqual(final["total_execution_fees"], (10 + 10 + 5) * 2)
        self.assertAlmostEqual(final["ending_cash"], 12000 + final["net_cumulative_pnl"] - 6000)
        self.assertAlmostEqual(final["free_cash"], final["ending_cash"] - final["required_margin"])

    def test_constant_fee_correction_changes_mean_but_not_known_variance(self):
        stages = result(160)["answers_by_turn"]
        self.assertEqual(stages["1"]["standard_error"], stages["2"]["standard_error"])
        self.assertAlmostEqual(stages["1"]["net_cost_improvement"] - stages["2"]["net_cost_improvement"], 0.5 * 4)
        self.assertLess(stages["4"]["standard_error"], stages["3"]["standard_error"])

    def test_execution_correction_changes_matching_not_retention_mandate(self):
        stages = result(161)["answers_by_turn"]
        self.assertEqual(stages["3"]["shares_sold"], 0)
        self.assertEqual(stages["4"]["shares_sold"], 60)
        self.assertGreaterEqual(stages["4"]["original_p_retained"], 30)
        self.assertEqual(stages["4"]["q_shares_bought"], 2*stages["4"]["shares_sold"])

    def test_marked_value_does_not_create_depth_or_cash(self):
        stages = result(162)["answers_by_turn"]
        self.assertEqual(stages["1"]["available_cash"], stages["2"]["available_cash"])
        self.assertEqual(stages["2"]["immediate_b_sale_proceeds"], 140*10)
        self.assertEqual(stages["4"]["shares_sold"][1], 5)
        self.assertEqual(stages["4"]["maximum_redemption"] + stages["4"]["unpaid_shortfall"], 25000)

    def test_information_policy_changes_when_loss_mandate_binds(self):
        stages = result(163)["answers_by_turn"]
        self.assertEqual(stages["3"]["conditional_policy"], ["B", "B"])
        self.assertFalse(stages["3"]["purchase_study"])
        self.assertEqual(stages["4"]["conditional_policy"], ["half_A", "B"])
        self.assertTrue(stages["4"]["purchase_study"])
        for record in stages.values():
            self.assertAlmostEqual(sum(record["signal_probabilities"]), 1)
            self.assertAlmostEqual(sum(p*q for p, q in zip(record["signal_probabilities"], record["posterior_good_probabilities"])), 0.4)

    def test_order_dedup_bust_and_cancel_terminal_state(self):
        packet = load_research_packet("q164")
        baseline = audit_case(packet)
        for turn in packet["turns"]:
            key = "events" if "events" in turn["inputs"] else "new_events"
            extra = [copy.deepcopy(row) for row in turn["inputs"][key] if row["event"] in ("fill", "fill_duplicate")]
            for row in extra:
                row["event"] = "fill_duplicate"
            turn["inputs"][key].extend(extra)
        self.assertEqual(audit_case(packet)["answers_by_turn"], baseline["answers_by_turn"])
        stages = baseline["answers_by_turn"]
        self.assertEqual([stages[str(t)]["next_action"] for t in range(1, 5)], ["WAIT", "CANCEL_O2_WAIT", "WAIT", "SUBMIT"])
        self.assertEqual(stages["4"]["confirmed_inventory"], 40 + 20 + 5)
        self.assertEqual(stages["4"]["new_order_quantity"] % 5, 0)
        self.assertLessEqual(stages["4"]["cash_required"], 1400)
        self.assertLessEqual(stages["4"]["inventory_if_filled"], 80)


if __name__ == "__main__":
    unittest.main()
