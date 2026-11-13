"""Public-contract and financial boundary checks without scorer/reference imports."""
import copy
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

from quality import decision_numerical as decision


ROOT = Path(__file__).resolve().parents[1]


class DecisionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = {cid: json.loads((ROOT / "questions/challenges" / (cid + ".json")).read_text())["question"]
                         for cid in decision.CHECKS}
        cls.derived = {cid: check(cls.questions[cid]) for cid, (check, _) in decision.CHECKS.items()}

    def test_all_forty_declared_outputs_and_exact_contract_scope(self):
        self.assertEqual(sum(len(result[0]) for result in self.derived.values()), 40)
        for cid, (answers, certificate) in self.derived.items():
            self.assertEqual(set(answers), set(self.questions[cid]["outputs"]))
            self.assertEqual(certificate["canonical_question_sha256"], decision.REVIEWED_QUESTIONS[cid])
            changed = copy.deepcopy(self.questions[cid])
            changed["sources"][0]["text"] += " Changed contract."
            with self.assertRaisesRegex(ValueError, "differs from reviewed input"):
                decision.CHECKS[cid][0](changed)

    def test_module_and_all_derivations_have_no_scorer_or_answer_import(self):
        script = """
import json,sys
from pathlib import Path
from quality.decision_numerical import CHECKS
for cid,(check,_) in CHECKS.items():
    check(json.loads(Path('questions/challenges',cid+'.json').read_text())['question'])
assert not any(name.startswith(('scoring_v2','scoring_v3','evals.','quality.inventory')) for name in sys.modules)
print('ok')
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_correlated_information_has_two_symmetric_optima(self):
        answers, certificate = self.derived["c205"]
        self.assertEqual(certificate["enumerated_policies"], 2 * (3 + 2 * 9) ** 2 + 3)
        self.assertEqual(len(certificate["all_optimal_policies"]), 2)
        self.assertEqual({policy[0] for policy in certificate["all_optimal_policies"]}, {"Q", "R"})
        self.assertEqual(answers["query_again_after"], ["negative"])
        self.assertEqual(answers["terminal_actions"], ["X", "X", "Y", "abstain"])
        self.assertEqual(answers["second_positive_probability"], .1 / (.1 + .4))
        self.assertEqual(answers["expected_query_cost"], 3 + .5 * 3)
        self.assertEqual(answers["optimal_net_value"], (.4 + .1 + .1) * 30 - 4.5)
        self.assertEqual(answers["information_value"], answers["optimal_net_value"] - answers["no_information_value"])

    def test_repeated_query_is_fixed_information_and_stopping_is_available(self):
        cells = [(F(1, 2), 0, {"A": 0}), (F(1, 2), 1, {"A": 1})]
        payoff = {"left": (10, 0), "right": (0, 10)}
        rows = decision.research_policies(cells, payoff, {"A": F(1)}, F(2), distinct=False)
        self.assertEqual(rows[0]["net"], 9)
        self.assertEqual(rows[0]["policy"][1], ("stop", "stop"))
        repeated = [row for row in rows if row["policy"][1] == ("A", "A")]
        self.assertEqual(max(row["net"] for row in repeated), 8)
        costly = decision.research_policies(cells, payoff, {"A": F(6)}, F(12), distinct=False)
        self.assertEqual(costly[0]["policy"][0], "none")
        self.assertEqual(costly[0]["net"], 5)

    def test_noisy_policy_uses_joint_cells_and_branch_fee(self):
        answers, certificate = self.derived["c217"]
        self.assertEqual(certificate["enumerated_policies"], 3 * (3 + 2 * 9) ** 2 + 3)
        self.assertEqual(len(certificate["all_optimal_policies"]), 1)
        self.assertEqual(answers["first_query"], "A")
        self.assertEqual(answers["second_queries"], ["stop", "B"])
        self.assertEqual(answers["terminal_actions"], ["pass", "pass", "hedge", "invest"])
        # Joint mass A=1 is48/100; A1,B0 pays hedge, A1,B1 pays invest.
        self.assertEqual(answers["expected_query_cost"], float(F(1, 2) + F(48, 100) / 2))
        self.assertEqual(answers["expected_gross_payoff"], float(F((3 + 8) * 10 - (5 + 3) * 8 + (9 + 16) * 30 - (3 + 1) * 40, 100)))
        self.assertEqual(answers["optimal_net_value"], float(F(636 - 74, 100)))

    def test_path_budget_and_lexicographic_ties(self):
        cells = [(F(1, 4), 0, {"A": left, "B": right}) for left, right in itertools.product((0, 1), repeat=2)]
        rows = decision.research_policies(cells, {"z": (1,), "a": (1,)}, {"A": F(0), "B": F(0)}, F(0))
        self.assertEqual(rows[0]["policy"], ("A", ("B", "B"), ("a",) * 4))
        constrained = decision.research_policies(cells, {"a": (1,)}, {"A": F(1), "B": F(2)}, F(2))
        self.assertTrue(all(row["policy"][1] == ("stop", "stop") for row in constrained))

    def test_noisy_policy_value_also_follows_backward_conditioning(self):
        good = [2, 3, 4, 5, 3, 8, 9, 16]
        bad = [20, 8, 7, 3, 5, 3, 3, 1]
        cells = [(F(weight, 100), state, bits) for state, row in enumerate((good, bad))
                 for weight, bits in zip(row, itertools.product((0, 1), repeat=3))]
        payoffs = ((30, -40), (10, -8), (0, 0))
        costs = (F(1, 2), F(1, 2), F(3, 2))

        def stop_value(subset):
            return max(sum(probability * payoff[state] for probability, state, _ in subset)
                       for payoff in payoffs)

        first_values = []
        for first in range(3):
            value = -costs[first]
            for bit in (0, 1):
                branch = [cell for cell in cells if cell[2][first] == bit]
                mass = sum(cell[0] for cell in branch)
                alternatives = [stop_value(branch)]
                for second in range(3):
                    if second == first or costs[first] + costs[second] > 4:
                        continue
                    alternatives.append(sum(stop_value([cell for cell in branch if cell[2][second] == observation])
                                            for observation in (0, 1)) - mass * costs[second])
                value += max(alternatives)
            first_values.append(value)
        self.assertEqual(first_values[0], F(281, 50))
        self.assertEqual(max(first_values + [stop_value(cells)]), F(281, 50))
        self.assertEqual(float(max(first_values)), self.derived["c217"][0]["optimal_net_value"])

    def test_exchange_order_bust_and_surviving_reservations(self):
        answers, certificate = self.derived["c206"]
        self.assertEqual(certificate["surviving_fill_quantities"], {"FS1": 30, "FS2": 20, "FB": 30})
        self.assertEqual(certificate["accepted_remaining"], {"S": 0, "B": 20})
        self.assertEqual(answers["current_inventory"], 120 - (40 + 20 - 10) + 30)
        self.assertEqual(answers["current_cash"], 1000 + (40 + 20 - 10) * 12 - 30 * 10)
        self.assertEqual(answers["available_inventory"], answers["current_inventory"] - 25 - 50)
        self.assertEqual(answers["reserved_buy_cash"], (50 - 30) * 10)
        requirement = certificate["cash_requirement"]
        self.assertGreaterEqual(answers["final_cash"], requirement)
        self.assertLess(certificate["one_less_sale_cash"], requirement)
        self.assertLess(certificate["next_lot_at_maximum_sale_cash"], requirement)

    def test_post_cancel_correction_does_not_reopen_and_duplicate_is_idempotent(self):
        orders = [{"id": "B", "side": "buy", "total": 80, "limit": 10}]
        fill = {"sequence": 1, "order": "B", "type": "fill", "fill_id": "F1", "quantity": 20, "price": 9, "day": 1}
        events = [fill, {"sequence": 2, "order": "B", "type": "replace", "total": 60, "limit": 11},
                  {"sequence": 3, "order": "B", "type": "cancel_ack"},
                  {"sequence": 4, "order": "B", "type": "correction", "fill_id": "F1", "quantity": 12, "price": 10, "day": 1}]
        expected = decision.replay_orders(orders, events)
        actual = decision.replay_orders(orders, list(reversed(events + [dict(fill)])))
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]["B"]["remaining"], 0)
        self.assertEqual(actual[1]["F1"]["quantity"], 12)
        self.assertEqual(actual[1]["F1"]["price"], 10)

    def test_common_repair_objective_and_distinct_runner_up(self):
        answers, certificate = self.derived["c213"]
        self.assertEqual(certificate["enumerated_actions"], 9 ** 3 * 6 * 5)
        self.assertEqual(answers["action"], [4, 4, 3, 0, 0])
        # Sale impacts are2,0,0; three issuer tickets; gross receipts booked separately.
        self.assertEqual(answers["implementation_cost"], 2 + 3)
        self.assertEqual(answers["early_cash"], 20 + 30 - 5 - 40)
        self.assertEqual(answers["scenario_cash"], [5 + 20 + 10 - 25 - 3] * 2)
        self.assertEqual(answers["objective"], float(F(5) + F(3, 10) * 25 + F(1, 5) * 6))
        next_state = decision.repair_state(certificate["next_best_action"])
        self.assertTrue(next_state["feasible"])
        self.assertNotEqual(certificate["next_best_action"], tuple(answers["action"]))
        self.assertEqual(answers["next_best_objective"], float(next_state["objective"]))

    def test_both_borrowing_bases_and_early_liquidity_are_required(self):
        one_base = decision.repair_state((2, 7, 1, 2, 1))
        self.assertEqual(one_base["slacks"]["borrowing_base_1"], 0)
        self.assertEqual(one_base["slacks"]["borrowing_base_2"], -1)
        self.assertTrue(all(value >= 0 for key, value in one_base["slacks"].items() if key != "borrowing_base_2"))
        self.assertFalse(one_base["feasible"])
        boundary = decision.repair_state((2, 3, 5, 3, 0))
        self.assertEqual(boundary["early"], 0)
        self.assertTrue(boundary["feasible"])
        late_only = decision.repair_state((2, 3, 5, 4, 0))
        self.assertLess(late_only["early"], 0)
        self.assertGreater(min(late_only["final"]), 0)
        self.assertFalse(late_only["feasible"])

    def test_negative_stress_loss_is_not_a_cash_receipt(self):
        state = decision.repair_state((1, 5, 4, 2, 0))
        self.assertEqual(state["losses"], (-1, 32))
        self.assertTrue(state["feasible"])
        before_stress = state["early"] + 10 + 4 - 3
        self.assertEqual(state["final"], (before_stress, before_stress - 32))

    def test_settlement_state_uses_corrected_trade_and_keeps_live_sell(self):
        answers, certificate = self.derived["c218"]
        self.assertEqual(certificate["surviving_fill_quantities"], {"F1": 12, "F2": 15, "F3": 5, "F4": 10})
        self.assertEqual(certificate["accepted_remaining"], {"B": 0, "S": 20})
        self.assertEqual(answers["owned_inventory_at_cutoff"], 70 + 12 + 15 + 5 - 10)
        self.assertEqual(answers["settled_inventory_at_cutoff"], 70)
        self.assertEqual(answers["old_trade_cash_by_day"], [-12 * 10 + 10 * 12, -(15 + 5) * 11])
        self.assertEqual(answers["inventory_after_T1"], 70 - 15 + 12 - 10 - 25)

    def test_final_inventory_objective_and_fee_then_lexicographic_ties(self):
        answers, certificate = self.derived["c218"]
        tied = certificate["maximum_inventory_minimum_fee_actions"]
        self.assertEqual(tied, [(15, 0, 0), (20, 5, 0), (25, 10, 0)])
        self.assertEqual(answers["action"], list(min(tied)))
        self.assertEqual({decision.settlement_repair(action)["final_owned"] for action in tied}, {52})
        higher_fee = decision.settlement_repair((25, 10, 100))
        self.assertTrue(higher_fee["feasible"])
        self.assertEqual(higher_fee["final_owned"], 52)
        self.assertGreater(higher_fee["fee"], answers["bridge_fee"])

    def test_exact_final_cash_boundary_and_unsettled_share_constraints(self):
        boundary = decision.settlement_repair((25, 10, 100))
        self.assertEqual(boundary["cash"][2], 100)
        self.assertTrue(boundary["feasible"])
        tightened = decision.settlement_repair((25, 10, 100), reserve=(50, 100 + F(1, 1000)))
        self.assertFalse(tightened["feasible"])
        unsettled_sale = decision.settlement_repair((55, 0, 0))
        self.assertLess(unsettled_sale["slacks"]["sale_resource"], 0)
        delivery_conflict = decision.settlement_repair((30, 0, 0))
        self.assertGreaterEqual(delivery_conflict["slacks"]["sale_resource"], 0)
        self.assertLess(delivery_conflict["slacks"]["inventory_T1"], 0)
        self.assertFalse(delivery_conflict["feasible"])


if __name__ == "__main__":
    unittest.main()
