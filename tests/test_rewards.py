"""Testes das recompensas calculadas a partir da simulação."""

from __future__ import annotations

import unittest

from autopilot.config import RewardConfig
from autopilot.environment import compute_interval_reward, reward_components


def snapshot(
    *,
    crossed: float = 0.0,
    wait: float = 0.0,
    queue_accum: float = 0.0,
    samples: float = 0.0,
    last_queue: float = 0.0,
) -> dict[str, float]:
    return {
        "crossed": crossed,
        "wait_seconds": wait,
        "queue_accum": queue_accum,
        "queue_samples": samples,
        "last_queue": last_queue,
    }


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RewardConfig(
            throughput_reward=1.0,
            waiting_penalty_per_vehicle_second=0.02,
            queue_penalty_per_vehicle=0.05,
            long_queue_threshold=12.0,
            long_queue_penalty=0.03,
            switch_penalty=2.0,
        )

    def test_throughput_is_positive(self) -> None:
        components = reward_components(
            snapshot(), snapshot(crossed=3), self.config
        )
        self.assertAlmostEqual(components["throughput"], 3.0)
        self.assertAlmostEqual(components["total"], 3.0)

    def test_waiting_and_queue_penalize(self) -> None:
        components = reward_components(
            snapshot(),
            snapshot(crossed=2, wait=50.0, queue_accum=30.0, samples=3.0),
            self.config,
        )
        self.assertAlmostEqual(components["wait_penalty"], 1.0)
        self.assertAlmostEqual(components["avg_queue"], 10.0)
        self.assertAlmostEqual(components["queue_penalty"], 0.5)
        self.assertAlmostEqual(components["long_queue_penalty"], 0.0)
        self.assertAlmostEqual(components["total"], 0.5)

    def test_long_queue_extra_penalty(self) -> None:
        components = reward_components(
            snapshot(),
            snapshot(crossed=0, queue_accum=45.0, samples=3.0),
            self.config,
        )
        self.assertAlmostEqual(components["avg_queue"], 15.0)
        self.assertAlmostEqual(components["long_queue_penalty"], 0.09)
        self.assertAlmostEqual(components["total"], -0.84)

    def test_more_waiting_lowers_reward(self) -> None:
        low = compute_interval_reward(
            snapshot(), snapshot(crossed=2, wait=10.0), self.config
        )
        high = compute_interval_reward(
            snapshot(), snapshot(crossed=2, wait=100.0), self.config
        )
        self.assertLess(high, low)

    def test_no_events_gives_zero(self) -> None:
        self.assertAlmostEqual(
            compute_interval_reward(snapshot(), snapshot(), self.config), 0.0
        )

    def test_queue_average_uses_interval(self) -> None:
        previous = snapshot(queue_accum=10.0, samples=5.0)
        current = snapshot(queue_accum=22.0, samples=9.0)
        components = reward_components(previous, current, self.config)
        self.assertAlmostEqual(components["avg_queue"], 3.0)

    def test_switch_penalty_is_configured(self) -> None:
        self.assertEqual(self.config.switch_penalty, 2.0)

    def test_negative_deltas_are_clamped(self) -> None:
        previous = snapshot(crossed=5, wait=10, queue_accum=10, samples=5)
        current = snapshot(crossed=4, wait=5, queue_accum=5, samples=4, last_queue=2)
        components = reward_components(previous, current, self.config)
        self.assertEqual(components["crossed"], 0.0)
        self.assertEqual(components["wait_seconds"], 0.0)
        self.assertLessEqual(components["total"], 0.0)


if __name__ == "__main__":
    unittest.main()
