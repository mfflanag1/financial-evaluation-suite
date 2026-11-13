"""Projection boundary controls for staged arithmetic audits."""
from copy import deepcopy
import json
import unittest

from quality import public_contracts as public


def poisoned(value):
    if type(value) is bool:
        return not value
    if type(value) in (int, float):
        return 987654321
    if isinstance(value, str):
        return 'POISONED_ANSWER'
    if isinstance(value, list):
        return ['POISONED_NESTED_ANSWER']
    if isinstance(value, dict):
        return {'poisoned': True}
    return None


class ProjectionBoundary(unittest.TestCase):
    def test_core_answer_values_and_evaluator_metadata_cannot_enter_projection(self):
        for track in ('challenges', 'variants'):
            cases = json.loads((public.ROOT / f'private_evaluator/{track}/cases.json').read_text())
            for case in cases:
                original = public.project_core(case)
                changed = deepcopy(case)
                for stage in changed['stages']:
                    stage['answers'] = {key: poisoned(value) for key, value in stage['answers'].items()}
                    for key in ('solution', 'checks', 'evidence_routes', 'wrong_answers', 'alternative_answers'):
                        stage[key] = 'POISONED_EVALUATOR_METADATA'
                self.assertEqual(original, public.project_core(changed), case['id'])

    def test_research_answers_and_solution_metadata_cannot_enter_projection(self):
        cases = json.loads((public.ROOT / 'private_evaluator/legacy/research58/evaluator.json').read_text())
        for case in cases:
            original = public.project_research(case)
            changed = deepcopy(case)
            for turn in changed['turns']:
                for key in ('answer', 'evidence', 'solution', 'checks', 'wrong_answers'):
                    turn[key] = 'POISONED_EVALUATOR_METADATA'
            self.assertEqual(original, public.project_research(changed), case['id'])

    def test_projection_does_not_alias_inputs(self):
        case = json.loads((public.ROOT / 'private_evaluator/challenges/cases.json').read_text())[0]
        expected = deepcopy(case)
        packet = public.project_core(case)
        packet['turns'][0]['inputs'].clear()
        packet['turns'][0]['sources'].clear()
        self.assertEqual(case, expected)

    def test_lookup_rejects_unrecognized_id(self):
        with self.assertRaises(ValueError):
            public.load_core_packet('unknown')
        with self.assertRaises(ValueError):
            public.load_workflow_packet('../outside')


if __name__ == '__main__':
    unittest.main()
