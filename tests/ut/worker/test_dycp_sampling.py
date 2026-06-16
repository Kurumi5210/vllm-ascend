import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vllm_ascend.worker.dycp_sampling import (
    iter_dycp_sample_seeds,
    make_dycp_sample_seed,
    temporary_dycp_sample_generators,
)


class TestDyCPSampling(unittest.TestCase):

    def test_make_dycp_sample_seed_is_stable(self):
        seed = make_dycp_sample_seed(1234, "req-a", 7, "target")

        self.assertEqual(seed, make_dycp_sample_seed(1234, "req-a", 7, "target"))
        self.assertNotEqual(seed, make_dycp_sample_seed(1234, "req-a", 8, "target"))
        self.assertNotEqual(seed, make_dycp_sample_seed(1234, "req-a", 7, "async_exponential"))
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 1 << 63)

    def test_iter_dycp_sample_seeds_only_selects_cp_random_requests(self):
        seeds = list(
            iter_dycp_sample_seeds(
                dycp_size=2,
                num_cp_request=3,
                req_ids=["req-a", "req-b", "req-c", "req-d"],
                num_tokens_no_spec=[5, 7, 11, 13],
                temperatures=[1.0, 0.0, 0.8, 1.0],
                existing_generator_indices={2},
                model_seed=1234,
                phase="target",
            )
        )

        self.assertEqual([seed.req_idx for seed in seeds], [0])
        self.assertEqual(
            seeds[0].seed,
            make_dycp_sample_seed(1234, "req-a", 5, "target"),
        )

    def test_temporary_dycp_sample_generators_are_reproducible_and_restored(self):
        def generator_factory(seed: int) -> random.Random:
            return random.Random(seed)

        generators_by_rank = []
        sampled_by_rank = []
        for _ in range(2):
            generators = {2: random.Random(9999)}
            with temporary_dycp_sample_generators(
                generators,
                generator_factory=generator_factory,
                dycp_size=2,
                num_cp_request=2,
                req_ids=["req-a", "req-b", "req-c"],
                num_tokens_no_spec=[5, 7, 11],
                temperatures=[1.0, 0.8, 1.0],
                model_seed=1234,
                phase="target",
            ):
                sampled_by_rank.append([generators[i].random() for i in range(2)])
                self.assertIn(2, generators)
                self.assertNotIn(3, generators)

            generators_by_rank.append(generators)

        self.assertEqual(sampled_by_rank[0], sampled_by_rank[1])
        self.assertEqual(set(generators_by_rank[0]), {2})
        self.assertEqual(set(generators_by_rank[1]), {2})

    def test_temporary_dycp_sample_generators_skip_non_dycp(self):
        generators = {}

        with temporary_dycp_sample_generators(
            generators,
            generator_factory=random.Random,
            dycp_size=1,
            num_cp_request=2,
            req_ids=["req-a", "req-b"],
            num_tokens_no_spec=[5, 7],
            temperatures=[1.0, 0.8],
            model_seed=1234,
            phase="target",
        ):
            self.assertEqual(generators, {})

        self.assertEqual(generators, {})


if __name__ == "__main__":
    unittest.main()
