"""Testes de determinismo do ambiente e da avaliação justa."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from autopilot.agent import QLearningAgent
from autopilot.config import Config
from autopilot.environment import TrafficEnvironment
from autopilot.evaluation import compare_controllers, run_episode


class TestEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def run_episode(self, mode: str, *, agent: QLearningAgent | None = None) -> dict:
        env = TrafficEnvironment(self.config, agent=agent)
        env.reset(seed=99, mode=mode, training=False, episode_seconds=45.0)
        env.run()
        return env.final_metrics()

    def test_same_seed_reproduces_traditional_metrics(self) -> None:
        first = self.run_episode("traditional")
        second = self.run_episode("traditional")
        self.assertEqual(first, second)

    def test_same_seed_reproduces_greedy_ai_metrics(self) -> None:
        agent = QLearningAgent(self.config.agent, seed=3)
        first = self.run_episode("ai", agent=agent)
        second = self.run_episode("ai", agent=agent)
        self.assertEqual(first, second)

    def test_modes_receive_identical_arrivals(self) -> None:
        traditional = self.run_episode("traditional")
        ai = self.run_episode("ai", agent=QLearningAgent(self.config.agent, seed=3))
        # A agenda de chegadas é idêntica. O número efetivamente inserido na
        # via pode variar porque o spawn respeita o espaço livre da fila.
        self.assertEqual(traditional["arrived"], ai["arrived"])
        self.assertLessEqual(ai["spawned"], ai["arrived"])
        self.assertEqual(traditional["duration"], ai["duration"])

    def test_observation_is_discrete_and_in_range(self) -> None:
        env = TrafficEnvironment(self.config)
        env.reset(seed=5, mode="ai", training=False, episode_seconds=30.0)
        env.run()
        observation = env.observation()
        self.assertEqual(len(observation), 6)
        for value, upper in zip(observation, (3, 3, 3, 1, 1, 3)):
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, upper)


class TestEvaluationFairness(unittest.TestCase):
    def test_comparison_uses_same_seeds_and_never_updates_the_agent(self) -> None:
        config = Config()
        agent = QLearningAgent(config.agent, seed=11)
        # Um valor sentinela permite detectar qualquer atualização da tabela.
        agent.q_table[0, 0, 0, 0, 1, 0, 0] = 1.25
        before_table = agent.q_table.copy()
        before_counters = (
            agent.episodes_trained,
            agent.total_decisions,
            agent.total_updates,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.chart_path = root / "evaluation.png"
            config.report_path = root / "evaluation.json"
            result = compare_controllers(
                config,
                agent=agent,
                seeds=(701, 702),
                seconds=30.0,
                verbose=False,
                save_chart=False,
            )
            self.assertTrue(config.report_path.exists())

        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["seeds"], [701, 702])
        self.assertEqual(
            result["agent"],
            {
                "episodes_trained": before_counters[0],
                "epsilon_saved": agent.epsilon,
                "epsilon_used": 0.0,
                "exploration": False,
                "q_table_updates_during_eval": 0,
                "decisions_during_eval": 0,
                "q_table_unchanged": True,
            },
        )
        np.testing.assert_array_equal(agent.q_table, before_table)
        self.assertEqual(
            (
                agent.episodes_trained,
                agent.total_decisions,
                agent.total_updates,
            ),
            before_counters,
        )

        traditional = result["episodes_detail"]["traditional"]
        ai = result["episodes_detail"]["ai"]
        self.assertEqual(len(traditional), 2)
        self.assertEqual(len(ai), 2)
        for trad_row, ai_row in zip(traditional, ai):
            self.assertEqual(trad_row["seed"], ai_row["seed"])
            self.assertEqual(trad_row["arrived"], ai_row["arrived"])
            self.assertEqual(trad_row["duration"], ai_row["duration"])
            self.assertEqual(ai_row["epsilon_used"], 0.0)

    def test_single_episode_uses_requested_duration(self) -> None:
        config = Config()
        metrics = run_episode(config, seed=12, mode="traditional", seconds=20.0)
        self.assertAlmostEqual(metrics["duration"], 20.0, places=6)
        self.assertEqual(metrics["mode"], "traditional")


if __name__ == "__main__":
    unittest.main()
