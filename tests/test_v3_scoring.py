"""Regression reproductions for semantic integers and source citation duplicates."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scoring_v2 import engine as v2_engine, research as v2_research, source as v2_source
from scoring_v2.grade import grade_response as v2_grade
from scoring_v3 import VERSION, constraints, engine, research, sessions, source, workflow
from scoring_v3.common import FROZEN, ROOT, binding, catalog, read, verify
from scoring_v3.grade import grade_response, public_question
from scoring_v3.static import quant_score


def core_response(stage, cid, turn=1):
    return {'question_id': cid, 'turn': turn, 'answers': deepcopy(stage['answers']),
            'evidence': {k: deepcopy(v[0]) for k, v in stage['evidence_routes'].items()}}


def change_first(value, selector, replace):
    if not selector:
        return replace(value)
    result = deepcopy(value)
    index = 0 if selector[0] == '*' else selector[0]
    result[index] = change_first(result[index], selector[1:], replace)
    return result


class SemanticIntegerCorrections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = catalog()

    def core(self, cid):
        state = engine.new_session(self.bank[cid][1], 'supplied')
        engine.public_view(state)
        return state, engine.stage(state)

    def test_six_predecessor_defects_are_reproduced_and_corrected(self):
        for cid, field in [('c201', 'trade_quantities'), ('c201', 'optimal_plan_count'),
                           ('c202', 'independent_count'), ('c202', 'research_hypotheses')]:
            state, stage = self.core(cid)
            response = core_response(stage, cid)
            selector = constraints.selectors(cid, 1, field)[0]
            response['answers'][field] = change_first(response['answers'][field], selector, lambda v: v + 5e-7)
            self.assertTrue(v2_engine.score(stage, response, cid, 1, state['seen'])['all_supported_correct'])
            result = engine.score(stage, response, cid, 1, state['seen'])
            self.assertFalse(result['all_values_correct'])
            self.assertFalse(result['response_contract_valid'])
        for cid, turn, field in [('q138', 2, 'delivery_units'), ('q149', 1, 'eligible_issue_count')]:
            case = self.bank[cid][1]
            response = research.reference_response(case, turn)
            selector = constraints.selectors(cid, turn, field)[0]
            response['answers'][field] = change_first(response['answers'][field], selector, lambda v: v + 5e-7)
            self.assertTrue(v2_research.grade_turn(case, turn, response)['all_correct'])
            self.assertFalse(research.grade_turn(case, turn, response)['all_correct'])

    def test_all_approved_selectors_reject_fraction_bool_nonfinite_and_wrong_large_int(self):
        # Gate every approved leaf independently of its track's aggregation.
        for (cid, turn, field), selectors in constraints.APPROVED.items():
            track, case = self.bank[cid]
            if track == 'source':
                refs = source.load_case(FROZEN / 'legacy/source_grounded', cid)[1]['fields']
                reference = next(f['expected'] for f in refs if f['id'] == field)
            elif track in ('quant100', 'workflows'):
                reference = case['answer'][field]
            elif track == 'research58':
                reference = case['turns'][turn - 1]['answer'][field]
            else:
                reference = case['stages'][turn - 1]['answers'][field]
            self.assertTrue(constraints.exact(cid, turn, field, reference, reference), (cid, turn, field))
            for selector in selectors:
                integral = change_first(reference, selector, float)
                self.assertTrue(constraints.exact(cid, turn, field, integral, reference), (cid, turn, field))
                for replacement in [lambda v: v + 5e-7, lambda v: True,
                                    lambda v: 10**400, lambda v: float('nan'), lambda v: float('inf')]:
                    changed = change_first(reference, selector, replacement)
                    self.assertFalse(constraints.exact(cid, turn, field, changed, reference), (cid, turn, field))

    def test_every_approved_map_entry_is_enforced_by_its_actual_track(self):
        core_stages = {}
        for cid, (track, case) in self.bank.items():
            if track not in ('challenges', 'variants'):
                continue
            state = engine.new_session(case, 'supplied')
            while not state['finished']:
                engine.public_view(state)
                stage, turn = engine.stage(state), state['turn']
                core_stages[(cid, turn)] = (deepcopy(stage), deepcopy(state['seen']))
                engine.step(state, core_response(stage, cid, turn))
        for (cid, turn, field), selectors in constraints.APPROVED.items():
            track, case = self.bank[cid]
            if track == 'source':
                controls = read(FROZEN / 'legacy/source_grounded/controls' / f'{cid}.json')
                response = deepcopy(next(c['response'] for c in controls if c['expected_automated_pass']))
                from financial_evals.core import normalized
                for answer in response['answers'].values():
                    unique, seen = [], set()
                    for cite in answer.get('evidence', []):
                        key = (cite['source_id'], normalized(cite['quote']))
                        if key not in seen:
                            unique.append(cite)
                            seen.add(key)
                    answer['evidence'] = unique
                def grade(value):
                    result = grade_response(cid, value)
                    row = next(r for r in result['fields'] if r['id'] == field)
                    return result['automated_pass'], row['value_correct']
            elif track in ('quant100', 'workflows'):
                response = {'question_id': cid, 'answers': deepcopy(case['answer'])}
                def grade(value):
                    result = grade_response(cid, value)
                    row = next(r for r in result['fields'] if r['name'] == field)
                    return result['all_correct'], row['correct']
            elif track == 'research58':
                response = research.reference_response(case, turn)
                def grade(value):
                    result = grade_response(cid, value, turn=turn)
                    row = next(r for r in result['fields'] if r['name'] == field)
                    return result['all_correct'], row['value_correct']
            else:
                stage, seen = core_stages[(cid, turn)]
                response = core_response(stage, cid, turn)
                def grade(value):
                    result = engine.score(stage, value, cid, turn, seen)
                    row = next(r for r in result['fields'] if r['name'] == field)
                    return result['all_supported_correct'], row['value_correct']
            self.assertEqual(grade(response), (True, True), (cid, turn, field))
            for selector in selectors:
                for replacement, expected in [(float, (True, True)),
                                               (lambda v: v + 5e-7, (False, False)),
                                               (lambda v: True, (False, False)),
                                               (lambda v: 10**400, (False, False)),
                                               (lambda v: float('nan'), (False, False)),
                                               (lambda v: float('inf'), (False, False))]:
                    changed = deepcopy(response)
                    if track == 'source':
                        value = changed['answers'][field]['value']
                        changed['answers'][field]['value'] = change_first(value, selector, replacement)
                    else:
                        changed['answers'][field] = change_first(changed['answers'][field], selector, replacement)
                    self.assertEqual(grade(changed), expected, (cid, turn, field))

    def test_no_invented_integer_magnitude_limit_or_rounding(self):
        self.assertTrue(constraints.integral(10**400))
        self.assertFalse(constraints.integral(True))
        self.assertFalse(constraints.integral(8.0000005))
        self.assertTrue(constraints.integral(8.0))
        self.assertTrue(constraints.exact('q076', 1, 'min_months', 10**400, 10**400))

    def test_exact_integer_values_override_wide_tolerance(self):
        case = deepcopy(self.bank['q076'][1])
        field = next(f for f in case['outputs'] if f['name'] == 'min_months')
        field.update(atol=100, rtol=1)
        response = {'question_id': case['id'], 'answers': deepcopy(case['answer'])}
        response['answers']['min_months'] += 1
        self.assertFalse(quant_score(case, response)['all_correct'])

    def test_scalar_array_and_nested_mixed_selectors(self):
        self.assertTrue(constraints.valid('c224', 1, 'action', ['BUY', 2.0]))
        self.assertFalse(constraints.valid('c224', 1, 'action', ['BUY', 2.0000005]))
        self.assertTrue(constraints.valid('c220', 1, 'diagnostics', [1.25, 2.75, [0.0, 1.0, 1]]))
        self.assertFalse(constraints.valid('c220', 1, 'diagnostics', [1.25, 2.75, [0, True, 1]]))
        self.assertTrue(constraints.valid('q162', 3, 'shares_sold', [2.0, .25]))
        self.assertFalse(constraints.valid('q162', 3, 'shares_sold', [2.25, 2.0]))
        self.assertEqual(constraints.normalize('c224', 1, 'action', ['BUY', 2.0]), ['BUY', 2])
        normalized = constraints.normalize('q162', 3, 'shares_sold', [2.0, .25])
        self.assertIs(type(normalized[0]), int)
        self.assertIs(type(normalized[1]), float)
        for malformed in [None, {}, [], ['BUY'], ['BUY', []]]:
            self.assertFalse(constraints.valid('c224', 1, 'action', malformed))

    def test_continuous_siblings_retain_tolerances(self):
        case = self.bank['q162'][1]
        response = research.reference_response(case, 3)
        response['answers']['shares_sold'][1] += 5e-7
        self.assertTrue(v2_research.grade_turn(case, 3, response)['all_correct'])
        self.assertTrue(research.grade_turn(case, 3, response)['all_correct'])
        for cid, field in [('q085', 'q_max_shares'), ('q089', 'equivalent_ewma_half_life')]:
            self.assertEqual(constraints.selectors(cid, 1, field), [])
            case = self.bank[cid][1]
            self.assertTrue(grade_response(cid, {'question_id': cid, 'answers': case['answer']})['all_correct'])

    def test_integer_rules_are_stage_aware(self):
        # No restriction is inferred for unspecified fields or future stages.
        self.assertEqual(constraints.selectors('q162', 1, 'shares_sold'), [])
        self.assertEqual(constraints.selectors('q138', 2, 'delivery_units'), [['*']])
        self.assertEqual(constraints.selectors('q162', 3, 'shares_sold'), [[0]])

    def test_all_quant_workflow_and_research_references(self):
        for cid, (track, case) in self.bank.items():
            if track == 'quant100':
                response = {'question_id': cid, 'answers': case['answer']}
                self.assertTrue(grade_response(cid, response)['all_correct'], cid)
                for wrong in case['wrong_answers']:
                    bad = dict(response, answers=dict(case['answer'], **wrong['answers']))
                    self.assertFalse(grade_response(cid, bad)['all_correct'], cid)
            elif track == 'workflows':
                for variant in [case] + case['variants']:
                    response = {'question_id': cid, 'answers': variant['answer']}
                    self.assertTrue(grade_response(cid, response, variant=variant.get('name'))['all_correct'], cid)
            elif track == 'research58':
                responses = []
                for turn, stage in enumerate(case['turns'], 1):
                    response = research.reference_response(case, turn)
                    responses.append(response)
                    self.assertTrue(research.grade_turn(case, turn, response)['all_correct'], (cid, turn))
                    for wrong in stage['wrong_answers']:
                        bad = deepcopy(response)
                        bad['answers'].update(wrong['answers'])
                        self.assertFalse(research.grade_turn(case, turn, bad)['all_correct'], (cid, turn))
                self.assertTrue(research.grade_episode(case, responses)['all_turns_correct'], cid)

    def test_all_core_paths_alternatives_and_negative_controls(self):
        for cid, (track, case) in self.bank.items():
            if track not in ('challenges', 'variants'):
                continue
            session = engine.new_session(case, 'supplied')
            while not session['finished']:
                engine.public_view(session)
                stage, turn = engine.stage(session), session['turn']
                response = core_response(stage, cid, turn)
                for alternative in stage.get('alternative_answers', []):
                    result = engine.score(stage, dict(response, answers=alternative), cid, turn, session['seen'])
                    self.assertTrue(result['all_supported_correct'], (cid, turn))
                for wrong in stage['wrong_answers']:
                    bad = dict(response, answers=dict(response['answers'], **wrong['answers']))
                    self.assertFalse(engine.score(stage, bad, cid, turn, session['seen'])['all_values_correct'])
                engine.step(session, response)
            result = engine.report(session)
            self.assertTrue(result['all_supported_correct'], cid)
            self.assertTrue(result['response_contract_valid'], cid)
            self.assertIsNot(result['terminal_state_pass'], False, cid)

    def test_integral_action_forms_execute_identically(self):
        for cid, field in [('c207', 'trade_lots'), ('c212', 'quantity'), ('c224', 'action')]:
            initial, stage = self.core(cid)
            reference = core_response(stage, cid)
            alternate = deepcopy(reference)
            selectors = constraints.selectors(cid, 1, field)
            for selector in selectors:
                alternate['answers'][field] = change_first(alternate['answers'][field], selector, float)
            first, second = deepcopy(initial), deepcopy(initial)
            engine.step(first, reference)
            engine.step(second, alternate)
            self.assertEqual(first['state'], second['state'], cid)
            self.assertEqual(first['actions'], second['actions'], cid)
            self.assertTrue(second['scores'][0]['all_supported_correct'], cid)


    def test_fractional_mixed_action_does_not_execute(self):
        initial, stage = self.core('c224')
        response = core_response(stage, 'c224')
        response['answers']['action'][1] += 5e-7
        engine.step(initial, response)
        self.assertFalse(initial['scores'][0]['response_contract_valid'])
        self.assertFalse(initial['scores'][0]['all_values_correct'])
        self.assertEqual(initial['actions'][0]['status'], 'rejected')
        self.assertFalse(initial['economic_outcomes'][0]['accepted'])


class CitationCorrections(unittest.TestCase):
    root = FROZEN / 'legacy/source_grounded'

    def positive(self, cid):
        rows = read(self.root / 'controls' / f'{cid}.json')
        return deepcopy(next(r['response'] for r in rows if r['expected_automated_pass']))

    def test_identical_and_whitespace_equivalent_duplicate_reproductions(self):
        cid = 'credit-001'
        original = self.positive(cid)
        field = next(k for k, v in original['answers'].items() if v.get('evidence'))
        self.assertTrue(source.grade(self.root, cid, original)['automated_pass'])
        for whitespace in [False, True]:
            response = deepcopy(original)
            duplicate = deepcopy(response['answers'][field]['evidence'][0])
            if whitespace:
                duplicate['quote'] = '\n' + '  '.join(duplicate['quote'].split()) + '\n'
            response['answers'][field]['evidence'].append(duplicate)
            self.assertTrue(v2_source.grade(self.root, cid, response)['automated_pass'])
            result = source.grade(self.root, cid, response)
            self.assertFalse(result['automated_pass'])
            self.assertIn(field, result['duplicate_citation_fields'])
            result_field = next(f for f in result['fields'] if f['id'] == field)
            self.assertTrue(result_field['value_correct'])
            self.assertFalse(result_field['evidence_valid'])
            self.assertIn('duplicate_citation', result_field['failures'])

    def test_different_quotes_from_same_source_remain_permitted(self):
        cid = 'credit-001'
        response = self.positive(cid)
        _, _, sources = source.load_case(self.root, cid)
        field = next(k for k, v in response['answers'].items() if v.get('evidence'))
        sid = response['answers'][field]['evidence'][0]['source_id']
        existing = {v['quote'] for v in response['answers'][field]['evidence']}
        text = sources[sid]['text']
        quote = next(line.strip() for line in text.splitlines() if line.strip() and line.strip() not in existing)
        response['answers'][field]['evidence'].append({'source_id': sid, 'quote': quote})
        self.assertTrue(v2_source.grade(self.root, cid, response)['automated_pass'])
        self.assertTrue(source.grade(self.root, cid, response)['automated_pass'])

    def test_duplicates_do_not_convey_inherited_proof(self):
        # Find an existing reference that explicitly relies on a previous field.
        found = False
        for path in sorted((self.root / 'controls').glob('*.json')):
            for control in read(path):
                if not control['expected_automated_pass']:
                    continue
                response = deepcopy(control['response'])
                for downstream, answer in response['answers'].items():
                    dependencies = answer.get('support_refs', [])
                    if not dependencies:
                        continue
                    upstream = dependencies[0]
                    evidence = response['answers'][upstream].get('evidence', [])
                    if not evidence:
                        continue
                    evidence.append(deepcopy(evidence[0]))
                    result = source.grade(self.root, path.stem, response)
                    row = next(f for f in result['fields'] if f['id'] == downstream)
                    self.assertFalse(row['evidence_valid'])
                    found = True
                    break
                if found:
                    break
            if found:
                break
        self.assertTrue(found, 'Expected at least one inherited-proof reference')

    def test_all_retained_source_controls_and_input_immutability(self):
        changes = []
        for path in sorted((self.root / 'controls').glob('*.json')):
            for control in read(path):
                before = deepcopy(control['response'])
                result = source.grade(self.root, path.stem, control['response'])
                malformed = bool(result['duplicate_citation_fields'] or result['integer_constraint_failures'])
                self.assertEqual(result['automated_pass'], control['expected_automated_pass'] and not malformed,
                                 (path.stem, control['name']))
                self.assertEqual(control['response'], before)
                if control['expected_automated_pass'] and result['duplicate_citation_fields']:
                    # This is an explicitly repaired reference copy, never a
                    # candidate-response normalization or frozen-file edit.
                    from financial_evals.core import normalized
                    corrected = deepcopy(before)
                    for answer in corrected['answers'].values():
                        seen, unique = set(), []
                        for cite in answer.get('evidence', []):
                            key = (cite['source_id'], normalized(cite['quote']))
                            if key not in seen:
                                unique.append(cite)
                                seen.add(key)
                        answer['evidence'] = unique
                    self.assertTrue(source.grade(self.root, path.stem, corrected)['automated_pass'],
                                    (path.stem, control['name']))
                    changes.append((path.stem, control['name'], result['duplicate_citation_fields'],
                                    hashlib.sha256(json.dumps(corrected, sort_keys=True, separators=(',', ':'),
                                                              ensure_ascii=False, allow_nan=False).encode()).hexdigest()))
        manifest = read(ROOT / 'scoring_v3/reference_control_changes.json')
        recorded = [(r['case_id'], r['control_name'], r['duplicate_fields'], r['deduplicated_reference_sha256'])
                    for r in manifest['controls']]
        self.assertEqual(changes, recorded)
        self.assertEqual(len(changes), 27)

    def test_all_source_gold_values_pass_with_well_formed_unique_citations(self):
        for cid, (track, _) in catalog().items():
            if track != 'source':
                continue
            case, gold, sources = source.load_case(self.root, cid)
            expected = {f['id']: f['expected'] for f in gold['fields']}
            response = {'case_id': cid, 'answers': {}}
            for field in case['output_fields']:
                allowed = field.get('allowed_source_ids', case['source_ids'])
                response['answers'][field['id']] = {
                    'value': expected[field['id']],
                    'evidence': [{'source_id': sid, 'quote': sources[sid]['text']} for sid in allowed],
                }
            self.assertTrue(source.grade(self.root, cid, response)['automated_pass'], cid)


    def test_semantic_source_count_rejects_fraction_and_keeps_evidence_separate(self):
        cid, field = 'h2-008', 'oddlot_whole_kvue_shares'
        response = self.positive(cid)
        response['answers'][field]['value'] += 5e-7
        result = source.grade(self.root, cid, response)
        self.assertIn(field, result['integer_constraint_failures'])
        row = next(f for f in result['fields'] if f['id'] == field)
        self.assertFalse(row['value_correct'])
        self.assertTrue(row['evidence_valid'])


class VersionAndContractTests(unittest.TestCase):
    def test_frozen_predecessor_and_discrete_map_pins(self):
        verify()
        self.assertEqual(read(ROOT / 'scoring_v3/dependencies.json')['schema_version'], 1)
        self.assertIn('quality/discrete_fields.json', read(ROOT / 'scoring_v3/dependencies.json')['files'])
        self.assertNotEqual(binding(), __import__('scoring_v2.common', fromlist=['binding']).binding())

    def test_public_contracts_explain_selected_leaves_without_answers(self):
        source_packet = public_question('h2-008')
        self.assertEqual(source_packet['scorer_binding'], binding())
        self.assertIn('source_citation_rule', source_packet['scoring_corrections'])
        self.assertEqual(source_packet['scoring_corrections']['integer_fields']['oddlot_whole_kvue_shares'], [[]])
        for cid in ['q076', 'q162', 'q104', 'c220', 'c224']:
            packet = public_question(cid)
            self.assertEqual(packet['scorer_version'], VERSION)
            self.assertEqual(packet['scorer_binding'], binding())
            question = packet.get('question', packet)
            self.assertIn('integer_fields', question['scoring_corrections'])
            self.assertNotIn('expected', question)
            self.assertNotIn('answer', question)
            self.assertNotIn('reference_code', question)
        self.assertEqual(public_question('q101')['controller'], 'python -m scoring_v3.workflow')

    def test_sessions_reject_cross_version_states_on_every_entry(self):
        from scoring_v2 import sessions as old
        for cid in ['c207', 'q138']:
            predecessor = old.start(cid, 'supplied')
            for call in [lambda: sessions.public_view(predecessor), lambda: sessions.report(predecessor),
                         lambda: sessions.submit(predecessor, {}, 0)]:
                with self.assertRaisesRegex(ValueError, 'binding differs'):
                    call()
            current = sessions.start(cid, 'supplied')
            for call in [lambda: old.report(current), lambda: old.submit(current, {}, 0)]:
                with self.assertRaisesRegex(ValueError, 'binding differs'):
                    call()
            current['binding'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                sessions.public_view(current)

    def test_sequential_research_uses_corrected_grades(self):
        state = sessions.start('q138', 'supplied')
        case = catalog()['q138'][1]
        for turn in range(1, 5):
            response = research.reference_response(case, turn)
            if turn == 2:
                response['answers']['delivery_units'][0] += 5e-7
            sessions.public_view(state)
            sessions.submit(state, response, turn - 1)
        report = sessions.report(state)
        self.assertFalse(report['all_turns_correct'])
        self.assertEqual(report['scorer_version'], VERSION)
        self.assertFalse(report['turns'][1]['all_correct'])

    def test_imports_do_not_mutate_predecessor_in_same_or_fresh_process(self):
        script = '''
import json
from copy import deepcopy
from scoring_v2 import engine
from scoring_v2.common import catalog
c = catalog()['c202'][1]
s = engine.new_session(c, 'supplied'); engine.public_view(s); t = engine.stage(s)
r = {'question_id':'c202','turn':1,'answers':deepcopy(t['answers']), 'evidence':{k:v[0] for k,v in t['evidence_routes'].items()}}
r['answers']['independent_count'] += 5e-7
before = engine.score(t,r,'c202',1,s['seen'])
import scoring_v3.grade, scoring_v3.sessions, scoring_v3.workflow
after = engine.score(t,r,'c202',1,s['seen'])
print(json.dumps({'same':before==after,'baseline_accepts':after['all_supported_correct']}))
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), {'same': True, 'baseline_accepts': True})

    def test_sequential_cli_binds_state_and_exports_no_private_material(self):
        with tempfile.TemporaryDirectory() as temp:
            private, public = Path(temp) / 'private', Path(temp) / 'public'
            def run(*args, check=True):
                return subprocess.run([sys.executable, '-B', '-m', 'scoring_v3.sessions', *args], cwd=ROOT,
                                      capture_output=True, text=True, check=check)
            run('start', 'c207', '--session', str(private), '--out', str(public))
            state_path = private / 'state.json'
            state = read(state_path)
            state['private_test_sentinel'] = 'PRIVATE_STATE_MUST_NOT_APPEAR'
            state_path.write_text(json.dumps(state))
            while True:
                current = read(public / 'current.json')
                self.assertEqual(current['scorer_version'], VERSION)
                self.assertEqual(current['scorer_binding'], binding())
                encoded = json.dumps(current)
                for private_marker in ['PRIVATE_STATE_MUST_NOT_APPEAR', 'reference_code', 'evidence_routes', 'alternative_answers']:
                    self.assertNotIn(private_marker, encoded)
                if current.get('finished'):
                    break
                state = read(state_path)
                self.assertEqual(state['binding'], binding())
                stage = engine.stage(state['engine'])
                response = core_response(stage, 'c207', state['engine']['turn'])
                response_path = Path(temp) / 'response.json'
                response_path.write_text(json.dumps(response))
                run('submit', '--session', str(private), '--out', str(public),
                    '--revision', str(current['session_revision']), '--response', str(response_path))
            result = json.loads(run('report', '--session', str(private)).stdout)
            self.assertTrue(result['all_supported_correct'])
            self.assertEqual(result['scorer_binding'], binding())
            state = read(state_path)
            state['scorer_version'] = 'financial-reasoning-v2.0.0'
            state_path.write_text(json.dumps(state))
            result = run('resume', '--session', str(private), '--out', str(public), check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('binding differs', result.stderr)

    def test_cli_validate_export_and_grade(self):
        with tempfile.TemporaryDirectory() as temp:
            response = Path(temp) / 'response.json'
            case = catalog()['q076'][1]
            response.write_text(json.dumps({'question_id': 'q076', 'answers': case['answer']}))
            for args in [('validate',), ('export', 'q076'), ('grade', 'q076', str(response))]:
                result = subprocess.run([sys.executable, '-B', '-m', 'scoring_v3.grade', *args], cwd=ROOT,
                                        text=True, capture_output=True, check=True)
                self.assertEqual(json.loads(result.stdout)['scorer_version'], VERSION)


if __name__ == '__main__':
    unittest.main()
