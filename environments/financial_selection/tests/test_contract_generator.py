from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from financial_selection import reference
from financial_selection.contract import PacketError, input_digest, parse_public, strict_json
from financial_selection.generator import generate

KEY = bytes(range(32))  # Explicit test fixture; never a private evaluation key.


def rows(public, filename):
    return list(csv.DictReader(io.StringIO(public[filename])))


def replace_rows(public, filename, records):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    public[filename] = stream.getvalue()


class ContractGenerator(unittest.TestCase):
    def setUp(self):
        self.episode = generate(KEY, 'dev', 7)
        self.public = deepcopy(self.episode.public_files)

    def test_generation_is_keyed_reproducible_and_split_separated(self):
        self.assertEqual(self.episode, generate(KEY, 'dev', 7))
        for split, key, index in [('train', KEY, 7), ('final', KEY, 7), ('dev', bytes(reversed(KEY)), 7), ('dev', KEY, 8)]:
            other = generate(key, split, index)
            self.assertNotEqual(other.identity, self.episode.identity)
            self.assertNotEqual(other.public_files, self.public)

    def test_all_conditions_are_valid_and_reference_output_is_strict(self):
        for condition in ('mixed', 'positive', 'negative', 'unsupported', 'sparse'):
            packet = generate(KEY, 'train', 0, condition=condition).public_files
            parse_public(packet)
            result = strict_json(json.dumps(reference.solve(packet)))
            self.assertEqual(result['input_sha256'], input_digest(packet))
            if condition == 'negative':
                self.assertEqual(result['allocation'], [0, 0, 0, 0])
            if condition == 'positive':
                self.assertGreater(sum(result['allocation']), 0)
            if condition == 'unsupported':
                self.assertEqual([row['status'] for row in result['cells'][:2]], ['unsupported'] * 2)

    def test_archive_requires_zero_and_one_probability_consistency(self):
        for archived, numerator in [('0', '1'), ('1', '0')]:
            census = rows(self.public, 'universe.csv')
            census[0].update(archived=archived, pi_num=numerator, pi_den='1')
            changed = deepcopy(self.public)
            replace_rows(changed, 'universe.csv', census)
            with self.assertRaises(PacketError):
                parse_public(changed)

    def test_probability_components_bound_rational_growth(self):
        census = rows(self.public, 'universe.csv')
        census[0].update(pi_num='997', pi_den='1009')
        replace_rows(self.public, 'universe.csv', census)
        with self.assertRaises(PacketError):
            parse_public(self.public)

    def test_noninteger_json_and_duplicate_keys_rejected(self):
        for text in ('1.0', '1e309', 'NaN', 'Infinity', '{"x":1,"x":2}', '1' * 4097):
            with self.assertRaises(PacketError):
                strict_json(text)

    def test_malformed_utf8_and_csv_preserve_public_packet_error_type(self):
        for filename, text in [('contract.json', '\ud800'),
                               ('universe.csv', self.public['universe.csv'].splitlines()[0] + '\n' + 'x' * 140000 + ',1,2,0,0,1\n')]:
            changed = deepcopy(self.public)
            changed[filename] = text
            with self.assertRaises(PacketError):
                parse_public(changed)

    def test_malformed_container_rejected_as_packet_error(self):
        for field in ('regimes', 'audit_event_ids'):
            contract = json.loads(self.public['contract.json'])
            contract[field] = [{}]
            changed = deepcopy(self.public)
            changed['contract.json'] = json.dumps(contract)
            with self.assertRaises(PacketError):
                parse_public(changed)

    def test_unavailable_matured_feature_is_invalid(self):
        census = rows(self.public, 'universe.csv')
        event = next(row['event_id'] for row in census if int(row['label_maturity_tick']) < 1000)
        features = [row for row in rows(self.public, 'feature_releases.csv') if row['event_id'] != event]
        replace_rows(self.public, 'feature_releases.csv', features)
        with self.assertRaises(PacketError):
            parse_public(self.public)

    def test_exact_duplicate_releases_collapse_and_conflicts_fail(self):
        parsed = parse_public(self.public)
        features = rows(self.public, 'feature_releases.csv')
        features.append(dict(features[0]))
        replace_rows(self.public, 'feature_releases.csv', features)
        self.assertEqual(parse_public(self.public), parsed)
        features[-1]['signal'] = str(int(features[-1]['signal']) + 1)
        replace_rows(self.public, 'feature_releases.csv', features)
        with self.assertRaises(PacketError):
            parse_public(self.public)

    def test_reference_preserves_physical_crlf_identity(self):
        public = {name: text.replace('\n', '\r\n') for name, text in self.public.items()}
        with tempfile.TemporaryDirectory() as directory:
            for name, text in public.items():
                Path(directory, name).write_bytes(text.encode())
            subprocess.run([sys.executable, reference.__file__], cwd=directory, check=True,
                           capture_output=True, timeout=30)
            result = json.loads(Path(directory, 'policy.json').read_text())
        self.assertEqual(result, reference.solve(public))
        self.assertEqual(result['input_sha256'], input_digest(public))
        self.assertNotEqual(result['input_sha256'], input_digest(self.public))


if __name__ == '__main__':
    unittest.main()
