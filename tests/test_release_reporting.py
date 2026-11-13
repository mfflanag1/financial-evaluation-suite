"""Release summaries distinguish parent methods from subtest failure events."""
import io
import unittest

from scripts.check import ReleaseResult


def run_case(case):
    return unittest.TextTestRunner(stream=io.StringIO(), resultclass=ReleaseResult).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(case))


class ReleaseReporting(unittest.TestCase):
    def test_repeated_failed_subtests_do_not_create_negative_pass_counts(self):
        class Fixture(unittest.TestCase):
            def test_repeated(self):
                for index in range(3):
                    with self.subTest(index=index):
                        self.fail("controlled fixture")
            def test_success(self):
                pass
        result = run_case(Fixture)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(len(result.failures), 3)
        self.assertEqual(len(result.successful_methods), 1)

    def test_skipped_subtests_do_not_count_parent_as_passed(self):
        class Fixture(unittest.TestCase):
            def test_repeated(self):
                for index in range(2):
                    with self.subTest(index=index):
                        self.skipTest("controlled fixture")
        result = run_case(Fixture)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skipped), 2)
        self.assertEqual(result.successful_methods, [])

    def test_expected_failures_are_separate_from_passed_methods(self):
        class Fixture(unittest.TestCase):
            @unittest.expectedFailure
            def test_expected(self):
                self.fail("controlled fixture")
        result = run_case(Fixture)
        self.assertEqual(len(result.expectedFailures), 1)
        self.assertEqual(result.successful_methods, [])


if __name__ == "__main__":
    unittest.main()
