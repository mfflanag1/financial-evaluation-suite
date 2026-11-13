"""Audit coverage, numerical certificates, and preserved defect reproductions."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from quality import core_numerical, discrete, inventory, numerical
from quality.reproduce_defects import reproduce

ROOT=Path(__file__).resolve().parents[1]


class InventoryCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=inventory.independent_results()
        cls.bank=inventory.build_inventory(cls.evidence)
        cls.discrete=discrete.build()

    def test_all_cases_and_fields_have_scoped_status(self):
        self.assertEqual(self.bank["summary"]["main_cases"],224)
        self.assertEqual(self.bank["summary"]["related_variants"],6)
        self.assertEqual(self.bank["summary"]["main_output_fields"],1924)
        self.assertEqual(self.bank["summary"]["variant_output_fields"],48)
        self.assertEqual(len({case["id"] for case in self.bank["cases"]}),230)
        self.assertEqual(self.discrete["summary"]["stage_fields"],1972)
        self.assertEqual(self.discrete["summary"]["cases"],230)
        for case in self.bank["cases"]:
            self.assertIn(case["verification"]["independent_numerical_recomputation"],
                          ("matches_stored_reference","not_performed","reference_discrepancy"))
            self.assertEqual(case["verification"]["practitioner_review"],"not_documented")
            self.assertFalse(case["public_exposure"]["unseen_holdout_claim_supported"])
            self.assertEqual(case["public_exposure"]["reference_answers"],"published_in_repository")

    def test_catalog_reserved_labels_are_not_privacy_claims(self):
        reserved=[case for case in self.bank["cases"] if case["declared_split"]=="holdout"]
        self.assertEqual(len(reserved),8)
        for case in reserved:
            self.assertIsNone(case["input_artifact"])
            self.assertTrue((ROOT/case["evaluator_artifact"]["path"]).is_file())
            self.assertEqual(case["public_exposure"]["case_material"],"published_in_repository")

    def test_checked_in_inventory_matches_current_inputs(self):
        self.assertEqual(inventory.read(ROOT/"quality/inventory.json"),self.bank)
        self.assertTrue(inventory.evidence_equal(inventory.read(ROOT/"quality/independent_checks.json"),self.evidence))
        self.assertEqual(inventory.read(ROOT/"quality/discrete_fields.json"),self.discrete)

    def test_independent_calculations_do_not_take_answers_from_reference(self):
        original=inventory.read
        def corrupt_reference(path):
            document=original(path)
            if str(path).endswith("quant100/evaluator.json"):
                document=deepcopy(document)
                next(row for row in document if row["id"]=="q002")["answer"]["weight_a"]=0.9
            return document
        with patch.object(inventory,"read",side_effect=corrupt_reference):
            result=inventory.independent_results()
        check=next(row for row in result["checks"] if row["case_id"]=="q002")
        weight=next(row for row in check["fields"] if row["field"]=="weight_a")
        self.assertAlmostEqual(weight["derived"],0.125,places=7)
        self.assertEqual(weight["stored_reference"],0.9)
        self.assertFalse(weight["matches"])
        self.assertEqual(check["status"],"reference_discrepancy")

    def test_independent_cases_cover_every_declared_output(self):
        self.assertEqual(self.evidence["case_count"],230)
        self.assertEqual(self.evidence["matching_cases"],230)
        self.assertEqual(self.evidence["checked_fields"],1972)
        self.assertEqual({check["case_id"] for check in self.evidence["checks"] if check["case_id"].startswith("q")},
                         {f"q{i:03d}" for i in range(1,165)})
        schemas={row["id"]:{(stage["turn"],field["name"])
                            for stage in row["specification"]["stage_schemas"] for field in stage["outputs"]}
                 for row in self.bank["cases"]}
        for check in self.evidence["checks"]:
            self.assertEqual(schemas[check["case_id"]],{(field["turn"],field["field"]) for field in check["fields"]})
            if "retained_source_inputs" in check:
                packet=inventory.read(ROOT/check["input"]["path"])
                self.assertEqual(set(check["retained_source_inputs"]),set(packet["source_ids"]))
            self.assertTrue(all(field["matches"] for field in check["fields"]))

    def test_staged_reference_corruption_cannot_change_public_derivations(self):
        original=inventory.read
        def corrupt(path):
            document=original(path)
            if str(path).endswith("challenges/cases.json"):
                next(case for case in document if case["id"]=="c203")["stages"][0]["answers"]["action"]="POISONED"
            if str(path).endswith("research58/evaluator.json"):
                next(case for case in document if case["id"]=="q107")["turns"][0]["answer"]["reported_growth_pct"]=999
            return document
        with patch.object(inventory,"read",side_effect=corrupt):
            result=inventory.independent_results()
        for cid,field in (("c203","action"),("q107","reported_growth_pct")):
            check=next(row for row in result["checks"] if row["case_id"]==cid)
            row=next(row for row in check["fields"] if row["field"]==field)
            baseline=next(row for row in self.evidence["checks"] if row["case_id"]==cid)
            self.assertEqual(row["derived"],next(v["derived"] for v in baseline["fields"] if v["field"]==field))
            self.assertFalse(row["matches"])

    def test_mixed_arrays_have_specific_integer_selectors(self):
        lookup={(row["case_id"],row["turn"],row["field"]):row for row in self.discrete["fields"]}
        for turn in range(1,5):
            self.assertEqual(lookup[("c220",turn,"diagnostics")]["integer_selectors"],[[2,"*"]])
            self.assertEqual(lookup[("c224",turn,"action")]["integer_selectors"],[[1]])
        for turn in (3,4):
            self.assertEqual(lookup[("q162",turn,"shares_sold")]["integer_selectors"],[[0]])

    def test_quantity_names_do_not_automatically_create_integer_requirements(self):
        lookup={(row["case_id"],row["field"]):row for row in self.discrete["fields"]}
        for key in [("q085","q_max_shares"),("q089","equivalent_ewma_half_life"),("q054","stock_hedge"),("q121","remaining_conversion_shares")]:
            self.assertEqual(lookup[key]["status"],"explicit_exclusion")
            self.assertEqual(lookup[key]["integer_selectors"],[])
        self.assertEqual(lookup[("q013","base_units")]["status"],"approved_integer")
        self.assertEqual(lookup[("q020","bond_units")]["status"],"approved_integer")

    def test_reviewed_quantities_keep_contract_distinctions(self):
        lookup={(row["case_id"],row["turn"],row["field"]):row for row in self.discrete["fields"]}
        self.assertEqual(self.discrete["summary"]["by_status"]["approved_integer"],201)
        self.assertNotIn("candidate_requires_review",self.discrete["summary"]["by_status"])
        self.assertEqual(lookup[("h2-004",1,"export_credit_test_precedes_new_revolver_months")]["integer_selectors"],[[]])
        for turn in range(1,5):
            self.assertEqual(lookup[("q151",turn,"nominations")]["integer_selectors"],[["*"]])
            self.assertEqual(lookup[("q151",turn,"delivered_by_route")]["status"],"reviewed_no_additional_integer_constraint")
            self.assertEqual(lookup[("q157",turn,"pending_units")]["integer_selectors"],[])
        self.assertEqual(lookup[("q102",1,"ending_security_ledger")]["status"],"contract_clarification_needed")
        self.assertEqual(lookup[("c223",1,"action")]["status"],"reviewed_no_additional_integer_constraint")


class FinancialCertificates(unittest.TestCase):
    def packet(self,cid):
        return inventory.read(ROOT/f"questions/legacy/quant100/questions/{cid}.json")["inputs"]

    def test_minimum_variance_matches_stationarity_and_scales(self):
        base=[[0.04,0.005],[0.005,0.01]]
        weight,value,gradient=numerical.minimum_variance(base)
        self.assertAlmostEqual(weight,0.125,places=7)
        self.assertLess(abs(gradient),1e-8)
        scaled=[[entry*9 for entry in row] for row in base]
        second,second_value,_=numerical.minimum_variance(scaled)
        self.assertAlmostEqual(second,weight,places=7)
        self.assertAlmostEqual(second_value,9*value,places=12)

    def test_minimum_variance_finds_boundary_solution(self):
        weight,value,_=numerical.minimum_variance([[4,1.5],[1.5,1]])
        self.assertLess(weight,1e-7)
        self.assertAlmostEqual(value,1.0,places=10)

    def test_tail_atom_and_ties_conserve_probability_under_permutation(self):
        losses=[0,10,10,20];probabilities=[0.6,0.15,0.15,0.1]
        es,threshold,masses=numerical.tail_expectation(losses,probabilities,0.2)
        self.assertEqual(es,15)
        self.assertEqual(threshold,10)
        self.assertEqual(masses,[Fraction(0),Fraction(1,20),Fraction(1,20),Fraction(1,10)])
        order=[2,3,0,1]
        shuffled=numerical.tail_expectation([losses[i] for i in order],[probabilities[i] for i in order],0.2)
        self.assertEqual(shuffled[0],es)
        self.assertEqual(shuffled[2],[masses[i] for i in order])

    def test_tail_translation_and_all_probability_limit(self):
        losses=[-3,1,9];probabilities=[0.6,0.3,0.1]
        original=numerical.tail_expectation(losses,probabilities,0.25)[0]
        shifted=numerical.tail_expectation([x+7 for x in losses],probabilities,0.25)[0]
        self.assertEqual(shifted-original,7)
        full=numerical.tail_expectation(losses,probabilities,1)[0]
        self.assertEqual(full,sum(Fraction(str(p))*x for p,x in zip(probabilities,losses)))
        with self.assertRaises(ValueError):numerical.tail_expectation(losses,[0.6,0.3,0.2],0.1)

    def test_signed_desk_covariance_is_not_sum_of_standalone_vars(self):
        answers,evidence=numerical.q027(self.packet("q027"))
        self.assertEqual(answers["cross_variance"],16)
        self.assertEqual(evidence["portfolio_variance"],132)
        self.assertEqual(evidence["standalone_variance"],116)

    def test_parity_package_has_no_terminal_stock_exposure(self):
        answers,evidence=numerical.q051(self.packet("q051"))
        self.assertGreater(answers["initial_arbitrage_cash"],0)
        self.assertEqual(evidence["terminal_replication_errors"],[0.0]*4)
        reversed_quote=self.packet("q051");reversed_quote["call_quote"]=20
        reverse,_=numerical.q051(reversed_quote)
        self.assertEqual(reverse["package"],"sell_call_buy_put_long_stock")

    def test_call_implied_vol_reprices_and_incompatible_put_cannot_joint_fit(self):
        answers,evidence=numerical.q053(self.packet("q053"))
        self.assertLess(abs(evidence["call_repricing_error"]),1e-11)
        self.assertFalse(answers["joint_fit_possible"])
        consistent=self.packet("q053");consistent["put_quote"]=answers["consistent_put"]
        self.assertTrue(numerical.q053(consistent)[0]["joint_fit_possible"])

    def test_integer_hedge_is_cash_feasible_and_delta_neutral(self):
        answers,evidence=numerical.q054(self.packet("q054"))
        self.assertEqual(answers["contracts"],[10,1])
        self.assertEqual(evidence["residual_delta"],0)
        self.assertLessEqual(answers["cash_used"],4000)
        self.assertEqual(evidence["enumerated_choices"],169)

    def test_american_put_value_dominates_european_and_immediate_exercise(self):
        answers,evidence=numerical.q057(self.packet("q057"))
        self.assertTrue(evidence["dominates_immediate_and_european"])
        self.assertTrue(answers["exercise_first_down"])
        self.assertGreater(answers["exercise_premium"],0)

    def test_repaired_option_prices_satisfy_probability_and_martingale_constraints(self):
        answers,evidence=numerical.q060(self.packet("q060"))
        self.assertTrue(all(p>=0 for p in answers["probabilities"]))
        self.assertAlmostEqual(evidence["probability_sum"],1)
        self.assertAlmostEqual(evidence["martingale_mean"],102)
        self.assertGreater(evidence["objective_second_derivative"],0)
        self.assertFalse(answers["exact_fit"])

    def test_sharpe_minimum_track_record_is_smallest_feasible_integer(self):
        answers,evidence=numerical.q076(self.packet("q076"))
        self.assertEqual(answers["min_months"],51)
        self.assertLess(evidence["previous_month_statistic"],1.96)
        self.assertGreaterEqual(evidence["minimum_months_statistic"],1.96)
        self.assertTrue(evidence["annualization_preserves_statistic"])

    def test_factor_regression_residuals_are_orthogonal(self):
        _,evidence=numerical.q078(self.packet("q078"))
        self.assertLess(abs(evidence["residual_sum"]),1e-12)
        self.assertLess(abs(evidence["factor_residual_inner_product"]),1e-12)
        self.assertEqual(evidence["residual_degrees_of_freedom"],34)

    def test_hac_and_annual_covariance_use_different_horizons(self):
        answers,_=numerical.q079(self.packet("q079"))
        self.assertAlmostEqual(answers["long_run_variance_factor"],1.5)
        self.assertAlmostEqual(answers["annual_vol_autocorr_pct"]**2,127.5)
        iid=self.packet("q079");iid["autocorrelations"]={"beyond":0.0}
        simple,_=numerical.q079(iid)
        self.assertAlmostEqual(simple["t_hac"],simple["t_naive"])
        self.assertAlmostEqual(simple["annual_vol_autocorr_pct"],2.5*math.sqrt(12))

    def test_purge_includes_touching_label_end_and_embargo_follows_purge(self):
        answers,evidence=numerical.q083(self.packet("q083"))
        self.assertEqual(answers["n_train"],775)
        self.assertEqual(evidence["last_purged_after_test"],609)
        self.assertEqual(evidence["embargo_indices"],list(range(610,615)))
        self.assertTrue(evidence["no_overlap_after_removal"])

    def test_timing_shift_changes_results_on_identical_holding_periods(self):
        answers,evidence=numerical.q084(self.packet("q084"))
        self.assertEqual(evidence["holding_periods"],8)
        self.assertLess(answers["correct_cum_return_pct"],0)
        self.assertGreater(answers["lookahead_cum_return_pct"],0)

    def test_impact_budget_is_continuous_and_tight(self):
        answers,evidence=numerical.q085(self.packet("q085"))
        self.assertFalse(answers["q_max_shares"].is_integer())
        self.assertAlmostEqual(answers["impact_dollar_elasticity"],1.5,places=7)
        self.assertAlmostEqual(evidence["impact_at_maximum_bps"],40)
        self.assertGreater(evidence["impact_above_maximum_bps"],40)

    def test_multiple_trial_corrections_can_reverse_single_trial_conclusion(self):
        family,_=numerical.q080(self.packet("q080"))
        self.assertLess(family["p_single"],0.05)
        self.assertGreater(family["p_sidak"],0.05)
        sharpe,_=numerical.q087(self.packet("q087"))
        self.assertGreater(sharpe["psr_undeflated"],0.95)
        self.assertLess(sharpe["dsr_probability"],0.95)

    def test_discovery_rules_are_distinct(self):
        answers,_=numerical.q088(self.packet("q088"))
        self.assertEqual((answers["n_bh_discoveries"],answers["n_bonferroni"],answers["n_holm"]),(8,2,3))


    def test_independent_default_var_fails_subadditivity_for_stated_atoms(self):
        answers,evidence=numerical.q029(self.packet("q029"))
        self.assertEqual(answers["single_name_var"],0)
        self.assertEqual(answers["portfolio_var"],100)
        self.assertFalse(answers["var_subadditive_here"])
        self.assertAlmostEqual(evidence["joint_probability_sum"],1)

    def test_partial_identification_bounds_contain_completions(self):
        answers,_=numerical.q043(self.packet("q043"))
        lower,upper=answers["sharp_es_bounds"]
        self.assertEqual((lower,upper),(60,120))
        self.assertTrue(all(lower<=x<=upper for x in answers["completion_es"]))
        self.assertEqual(answers["limit_decision"],"not_identified")
        basket,_=numerical.q074(self.packet("q074"))
        self.assertLess(basket["net_value_bounds"][0],0)
        self.assertGreater(basket["net_value_bounds"][1],0)

    def test_common_regime_preserves_marginals_but_changes_tranche_loss(self):
        answers,evidence=numerical.q047(self.packet("q047"))
        self.assertEqual(evidence["default_states_enumerated"],8)
        self.assertAlmostEqual(evidence["probability_sum"],1)
        self.assertAlmostEqual(evidence["matched_marginal_pd"],0.066)
        self.assertNotAlmostEqual(answers["expected_layer_loss"],answers["independent_expected_layer_loss"])

    def test_same_moments_do_not_identify_tail_compliance(self):
        answers,evidence=numerical.q050(self.packet("q050"))
        for mean,variance in evidence["completion_mean_variance"]:
            self.assertAlmostEqual(mean,0)
            self.assertAlmostEqual(variance,100)
        self.assertLess(answers["completion_es"][0],50)
        self.assertGreater(answers["completion_es"][1],50)
        self.assertLess(answers["completion_vars"][1],0)

    def test_path_dependent_payoffs_do_not_collapse_to_terminal_spot(self):
        variance,evidence=numerical.q058(self.packet("q058"))
        self.assertTrue(evidence["same_initial_and_terminal"])
        self.assertNotEqual(*variance["realized_variances"])
        self.assertEqual(variance["call_payoff_difference"],0)
        barrier,evidence=numerical.q059(self.packet("q059"))
        self.assertFalse(barrier["terminal_only_replication"])
        self.assertAlmostEqual(evidence["probability_sum"],1)
        self.assertAlmostEqual(barrier["same_terminal_payoff_range"],5)

    def test_cds_event_accounting_and_richardson_overshoot(self):
        _,evidence=numerical.q071(self.packet("q071"))
        self.assertAlmostEqual(evidence["default_or_survival_probability_sum"],1)
        self.assertLess(abs(evidence["par_contract_net_pv"]),1e-8)
        digital,_=numerical.q075(self.packet("q075"))
        self.assertLess(digital["minimum_payoff"],0)
        self.assertGreater(digital["maximum_payoff"],1000)
        self.assertFalse(digital["bounded_on_menu"])

    def test_survival_weighting_uses_known_population_denominator(self):
        answers,evidence=numerical.q097(self.packet("q097"))
        self.assertEqual(evidence["known_population_size"],400)
        self.assertEqual(evidence["estimated_population_size"],300)
        self.assertNotEqual(answers["horvitz_thompson_mean"],answers["hajek_mean"])
        self.assertTrue(answers["sign_reversal"])

    def test_first_crossing_probability_conserves_mass_and_obeys_anytime_bound(self):
        answers,evidence=numerical.q098(self.packet("q098"))
        self.assertEqual(evidence["probability_conservation"],1)
        self.assertTrue(evidence["bound_satisfied"])
        self.assertGreater(answers["likelihood_stop_day"],answers["naive_fixed_test_stop_day"])
        self.assertGreaterEqual(answers["likelihood_at_stop"],20)

    def test_volatility_target_ignores_unavailable_realized_estimates(self):
        inputs=self.packet("q100")
        original,evidence=numerical.q100(inputs)
        inputs["realized_volatilities_revealed_after_period"]=[999,999,999]
        changed,_=numerical.q100(inputs)
        self.assertEqual(original,changed)
        self.assertEqual(original["opening_leverages"],[0.75,1.5,0.6])
        self.assertGreater(evidence["exit_fee"],0)
        self.assertTrue(any(row["interest"]<0 for row in evidence["period_ledger"]))

    def test_core_exact_search_and_joint_sign_protocol(self):
        packet=lambda cid:inventory.read(ROOT/f"questions/challenges/{cid}.json")["question"]
        repair,evidence=core_numerical.c201(packet("c201"))
        self.assertEqual(evidence["enumerated_holdings"],2401)
        self.assertEqual(repair["optimal_plan_count"],len(evidence["all_optimal_holdings"]))
        self.assertGreaterEqual(repair["early_cash"],0)
        self.assertGreaterEqual(repair["final_cash"],5)
        selected,evidence=core_numerical.c202(packet("c202"))
        self.assertTrue(evidence["disjoint_retained_intervals"])
        self.assertEqual(selected["research_hypotheses"],24)
        family,evidence=core_numerical.c214(packet("c214"))
        self.assertEqual(evidence["joint_sign_vectors"],64)
        self.assertGreaterEqual(family["selection_adjusted_p"],family["unadjusted_p"])


class PreservedDefectReproduction(unittest.TestCase):
    def test_v2_acceptance_of_fractional_semantic_integers_is_reproducible(self):
        report=reproduce()
        self.assertEqual(len(report["reproductions"]),6)
        for reproduction in report["reproductions"]:
            self.assertTrue(reproduction["valid_control_accepted"])
            self.assertTrue(reproduction["fractional_value_accepted"])
        # This passing reproduction test documents a preserved defect. It is
        # not a claim that the affected v2 scorer has been corrected.


if __name__=="__main__":unittest.main()
