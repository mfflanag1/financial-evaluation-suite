import copy
import json
import unittest
import information_variants as variants
import variant_engine as engine

class VariantIntegration(unittest.TestCase):
    def test_public_boundary_and_gold(self):
        for case in variants.build_cases():
            session=engine.new_session(case,'supplied')
            public=engine.public_view(session)
            self.assertEqual(len(public['question']['sources']),4)
            for private in ('information_parameters','diagnostic_role','variant_of','evidence_routes','solution','wrong_answers'):
                self.assertNotIn(private,json.dumps(public))
            stage=case['stages'][0]
            response={'question_id':case['id'],'turn':1,'answers':stage['answers'],'evidence':{k:v[0] for k,v in stage['evidence_routes'].items()}}
            engine.step(session,response)
            report=engine.report(session)
            self.assertTrue(report['all_supported_correct'])
            self.assertEqual(report['economic_outcomes'][0]['decision_regret'],0)
            self.assertIsNone(report['final_state'])
            self.assertEqual(report['actions'],[])

    def test_original_identity_controls_economics(self):
        case=variants.build_cases()[0];stage=case['stages'][0]
        session=engine.new_session(case,'supplied');engine.public_view(session)
        response={'question_id':'wrong','turn':1,'answers':stage['answers'],'evidence':{k:v[0] for k,v in stage['evidence_routes'].items()}}
        engine.step(session,response);report=engine.report(session)
        self.assertFalse(report['all_values_correct'])
        self.assertTrue(report['financial_content_correct'])
        self.assertFalse(report['economic_outcomes'][0]['action_feasible'])
        self.assertIsNone(report['economic_outcomes'][0]['decision_regret'])

    def test_binding_budget_not_reported_scalar(self):
        case=next(c for c in variants.build_cases() if c['id']=='c229')
        answer=copy.deepcopy(case['stages'][0]['answers'])
        ref=variants.enumerate_reference(variants.parameters())[0]
        answer.update(first_query=ref[1],second_queries=list(ref[2]),terminal_actions=list(ref[3]))
        session=engine.new_session(case,'supplied');engine.public_view(session)
        engine.step(session,{'question_id':case['id'],'turn':1,'answers':answer,'evidence':{}})
        outcome=engine.report(session)['economic_outcomes'][0]
        self.assertFalse(outcome['action_feasible']);self.assertIsNone(outcome['decision_regret'])

if __name__=='__main__':unittest.main()
