"""Regression reproductions and reference paths for the opt-in scorer."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scoring_v2 import engine, research, sessions, source
from scoring_v2.amendments import COUNT_FIELDS, RESEARCH_SUPPORT, quant_case
from scoring_v2.common import FROZEN, ROOT, catalog, verify
from scoring_v2.grade import grade_response, public_question
from financial_evals import core, quant
import challenge_engine as frozen_engine
import variant_engine as frozen_variant
from financial_evals import research as frozen_research


def response(stage, cid, turn=1):
    return {'question_id': cid, 'turn': turn, 'answers': deepcopy(stage['answers']),
            'evidence': {k: deepcopy(v[0]) for k, v in stage['evidence_routes'].items()}}


class CorrectedScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = catalog()

    def core_stage(self, cid):
        session = engine.new_session(self.bank[cid][1], 'supplied')
        engine.public_view(session)
        return session, engine.stage(session)

    def test_frozen_packages_unchanged(self):
        verify()

    def test_exact_integral_actions_execute_consistently(self):
        for cid, key in [('c207', 'trade_lots'), ('c212', 'quantity')]:
            base, stage = self.core_stage(cid)
            original = response(stage, cid)
            integral = deepcopy(original)
            value = integral['answers'][key]
            integral['answers'][key] = [float(v) for v in value] if isinstance(value, list) else float(value)
            frozen = deepcopy(base)
            frozen_engine.step(frozen, integral)
            self.assertTrue(frozen['scores'][0]['all_supported_correct'])
            self.assertEqual(frozen['actions'][0]['status'], 'rejected')
            normal, alternate = deepcopy(base), deepcopy(base)
            engine.step(normal, original)
            engine.step(alternate, integral)
            self.assertTrue(alternate['scores'][0]['all_supported_correct'])
            self.assertEqual(normal['state'], alternate['state'])
            self.assertEqual(normal['actions'], alternate['actions'])
            for bad in (0.0000005, True, 10**400):
                trial = deepcopy(base)
                r = deepcopy(original)
                if isinstance(value, list):
                    r['answers'][key][0] = value[0] + bad if type(bad) is float else bad
                else:
                    r['answers'][key] = value + bad if type(bad) is float else bad
                engine.step(trial, r)
                self.assertFalse(trial['scores'][0]['all_values_correct'])
                self.assertFalse(trial['scores'][0]['response_contract_valid'])
                self.assertEqual(trial['actions'][0]['status'], 'rejected')

    def test_count_fraction_reproductions(self):
        for cid, keys in COUNT_FIELDS.items():
            case = self.bank[cid][1]
            for key in keys:
                with self.subTest(cid=cid, field=key):
                    r = {'question_id': cid, 'answers': deepcopy(case['answer'])}
                    r['answers'][key] -= .5
                    self.assertTrue(quant.score(case, r)['all_correct'])
                    self.assertFalse(grade_response(cid, r)['all_correct'])
                    for bad in (True, str(case['answer'][key]), 10**400):
                        r['answers'][key] = bad
                        self.assertFalse(grade_response(cid, r)['all_correct'])
                    r['answers'][key] = float(case['answer'][key])
                    self.assertTrue(grade_response(cid, r)['all_correct'])

    def test_quant_menu_is_explicit(self):
        public = public_question('q056')
        self.assertEqual(public['inputs']['permitted_shares'], list(range(101)))
        self.assertEqual(self.bank['q056'][1]['inputs']['permitted_shares'], [0, 100])
        data = public['inputs']
        choices = []
        for shares in data['permitted_shares']:
            cash = max((data['calls_owed'] * max(spot - data['strike'], 0) - shares * bid)
                       / data['cash_growth'] for spot, bid in zip(data['terminal_spots'], data['terminal_stock_bids']))
            choices.append((shares * data['initial_stock_ask'] + cash, shares, cash))
        outlay, shares, cash = min(choices)
        self.assertEqual(shares, self.bank['q056'][1]['answer']['shares'])
        self.assertAlmostEqual(outlay, self.bank['q056'][1]['answer']['minimum_outlay'])
        self.assertAlmostEqual(cash, self.bank['q056'][1]['answer']['cash_account'])

    def test_all_quant_references_and_negative_controls(self):
        for cid, (track, case) in self.bank.items():
            if track != 'quant100':
                continue
            amended = quant_case(case)
            self.assertTrue(quant.score(amended, {'question_id': cid, 'answers': case['answer']})['all_correct'], cid)
            for wrong in case['wrong_answers']:
                self.assertFalse(quant.score(amended, {'question_id': cid,
                    'answers': dict(case['answer'], **wrong['answers'])})['all_correct'], cid)

    def test_all_core_and_diagnostic_reference_paths(self):
        for cid, (track, case) in self.bank.items():
            if track not in ('challenges', 'variants'):
                continue
            session = engine.new_session(case, 'supplied')
            while not session['finished']:
                engine.public_view(session)
                stage = engine.stage(session)
                r = response(stage, cid, session['turn'])
                for alternative in stage.get('alternative_answers', []):
                    self.assertTrue(engine.score(stage, dict(r, answers=alternative), cid,
                                                 session['turn'], session['seen'])['all_supported_correct'], cid)
                for wrong in stage['wrong_answers']:
                    bad = dict(r, answers=dict(r['answers'], **wrong['answers']))
                    self.assertFalse(engine.score(stage, bad, cid, session['turn'], session['seen'])['all_values_correct'], cid)
                engine.step(session, r)
            report = engine.report(session)
            self.assertTrue(report['all_supported_correct'], cid)
            self.assertTrue(report['response_contract_valid'], cid)
            self.assertIsNot(report['terminal_state_pass'], False, cid)

    def test_core_missing_fields_and_huge_integers(self):
        for cid, (track, case) in self.bank.items():
            if track not in ('challenges', 'variants'):
                continue
            session, stage = self.core_stage(cid)
            frozen = frozen_variant if track == 'variants' else frozen_engine
            r = response(stage, cid)
            empty = dict(r, answers={})
            self.assertTrue(frozen.score(stage, empty, cid, 1, session['seen'])['response_contract_valid'])
            self.assertFalse(engine.score(stage, empty, cid, 1, session['seen'])['response_contract_valid'])
            missing = deepcopy(r)
            del missing['evidence'][next(iter(missing['evidence']))]
            self.assertFalse(engine.score(stage, missing, cid, 1, session['seen'])['response_contract_valid'])
            key = next(k for k, v in stage['answers'].items() if type(v) in (int, float))
            r['answers'][key] = 10**400
            with self.assertRaises(OverflowError):
                frozen.score(stage, r, cid, 1, session['seen'])
            engine.step(session, r)
            self.assertFalse(session['scores'][0]['all_values_correct'], cid)
            self.assertFalse(session['scores'][0]['response_contract_valid'], cid)

    def test_known_core_citation_amendments(self):
        probes = [('c217', 'optimal_net_value', ['in_joint', 'in_actions', 'in_query']),
                  ('c224', 'transaction_cost', ['c224_mechanics', 'c224_round1']),
                  ('c227', 'expected_query_cost', ['in_query']),
                  ('c229', 'second_queries', ['in_query', 'in_format'])]
        for cid, key, route in probes:
            session, stage = self.core_stage(cid)
            r = response(stage, cid)
            r['evidence'][key] = route
            frozen = frozen_variant if cid in ('c227', 'c229') else frozen_engine
            self.assertFalse(frozen.score(stage, r, cid, 1, session['seen'])['all_supported_correct'])
            self.assertTrue(engine.score(stage, r, cid, 1, session['seen'])['all_supported_correct'])
            for bad in ([], route + ['unseen'], route + [route[0]]):
                r['evidence'][key] = bad
                self.assertFalse(engine.score(stage, r, cid, 1, session['seen'])['all_supported_correct'])
            r['evidence'][key] = route
            self.assertFalse(engine.score(stage, r, cid, 1, [])['all_supported_correct'])

    def test_research_citation_reproductions(self):
        for (cid, turn, key), extra in RESEARCH_SUPPORT.items():
            case = self.bank[cid][1]
            r = research.reference_response(case, turn)
            r['evidence'][key] = [extra] if cid == 'q139' else r['evidence'][key] + [extra]
            self.assertFalse(frozen_research.grade_turn(case, turn, r)['all_correct'], cid)
            self.assertTrue(research.grade_turn(case, turn, r)['all_correct'], cid)
            valid = r['evidence'][key][:]
            for bad in ([], valid + [extra], valid + ['unknown_packet']):
                r['evidence'][key] = bad
                self.assertFalse(research.grade_turn(case, turn, r)['all_correct'], cid)
            if cid != 'q139':
                r['evidence'][key] = [extra]
                self.assertFalse(research.grade_turn(case, turn, r)['all_correct'], cid)

    def test_all_research_references_controls_and_huge_values(self):
        for cid, (track, case) in self.bank.items():
            if track != 'research58':
                continue
            self.assertTrue(all(research.control_results(case)), cid)
            for t in case['turns']:
                r = research.reference_response(case, t['turn'])
                field = next(f for f in t['outputs'] if f['type'] in ('number', 'number_array'))
                if field['type'] == 'number_array':
                    r['answers'][field['name']][0] = 10**400
                else:
                    r['answers'][field['name']] = 10**400
                with self.assertRaises(OverflowError):
                    frozen_research.grade_turn(case, t['turn'], r)
                self.assertFalse(research.grade_turn(case, t['turn'], r)['all_correct'])

    def test_paypal_annual_summary_alternative(self):
        root = FROZEN / 'legacy/source_grounded'
        case, gold, sources = core.load_case(root, 'h2-001')
        r = {'case_id': 'h2-001', 'answers': {f['id']: {'value': f['expected'],
             'evidence': [{'source_id': sid, 'quote': s['text']} for sid, s in sources.items()]}
             for f in gold['fields']}}
        key = 'prior_year_non_gaap_income_recast_reduction'
        quotes = []
        for year in ('2023', '2024'):
            sid = 'h2a-paypal-' + year
            text = sources[sid]['text']
            start = text.index("FY'" + year[2:] + ' Financial Results')
            end = text.index('\f', start)
            quotes.append({'source_id': sid, 'quote': text[start:end]})
        text = sources['h2a-paypal-2024']['text']
        start = text.index('As announced on February 7, 2024')
        end = text.index('Prior period non-GAAP results have been recast to reflect this change.', start) + len('Prior period non-GAAP results have been recast to reflect this change.')
        quotes.append({'source_id': 'h2a-paypal-2024', 'quote': text[start:end]})
        r['answers'][key]['evidence'] = quotes
        self.assertFalse(core.grade(root, 'h2-001', r)['automated_pass'])
        self.assertTrue(source.grade(root, 'h2-001', r)['automated_pass'])
        self.assertFalse(core.grade(root, 'h2-001', r)['automated_pass'])
        self.assertEqual(source.load_case(root, 'h2-001')[1], source.load_case(root, 'h2-001')[1])
        separated = [{'source_id': 'h2a-paypal-2023', 'quote': quote}
                     for quote in ("FY'23 Financial Results", 'FY 2023', 'Non-GAAP', 'Net income', '$5,646')]
        r['answers'][key]['evidence'] = separated + quotes[1:]
        self.assertFalse(source.grade(root, 'h2-001', r)['automated_pass'])
        for bad in (quotes[:2], quotes[1:], [dict(quotes[0], quote='invented annual numbers')] + quotes[1:]):
            r['answers'][key]['evidence'] = bad
            self.assertFalse(source.grade(root, 'h2-001', r)['automated_pass'])

    def test_all_retained_source_controls(self):
        root = FROZEN / 'legacy/source_grounded'
        for path in (root / 'controls').glob('*.json'):
            for control in json.loads(path.read_text()):
                result = source.grade(root, path.stem, control['response'])
                self.assertEqual(result['automated_pass'], control['expected_automated_pass'],
                                 (path.stem, control['name']))

    def test_versioned_sessions_and_public_boundary(self):
        for cid in ('c207', 'q138'):
            state = sessions.start(cid, 'supplied')
            opening = sessions.public_view(state)
            self.assertNotIn('evidence_routes', json.dumps(opening))
            self.assertNotIn('solution', json.dumps(opening))
            self.assertNotIn('q138_t3_1', json.dumps(opening))
            if cid == 'c207':
                stage = engine.stage(state['engine'])
                r = response(stage, cid)
            else:
                r = research.reference_response(state['case'], 1)
            sessions.submit(state, r, 0)
            self.assertEqual(state['revision'], 1)
            with self.assertRaises(ValueError):
                sessions.submit(state, r, 0)
            state['binding'] = 'different'
            with self.assertRaises(ValueError):
                sessions.report(state)

    def test_cli_export_and_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'response.json'
            case = self.bank['q076'][1]
            path.write_text(json.dumps({'question_id': 'q076', 'answers': case['answer']}))
            run = subprocess.run([sys.executable, '-B', '-m', 'scoring_v2.grade', 'grade', 'q076', str(path)],
                                 cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            self.assertTrue(result['all_correct'])
            self.assertEqual(result['scorer_version'], 'financial-reasoning-v2.0.0')
        with self.assertRaises(ValueError):
            public_question('c219', turn=2)
        with self.assertRaises(ValueError):
            public_question('q138', turn=3)


if __name__ == '__main__':
    unittest.main()
