"""Testes do agente Q-Learning e da persistência da Q-table."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autopilot.agent import QLearningAgent
from autopilot.config import AgentConfig


class TestQLearningAgent(unittest.TestCase):
    def make_agent(self, **overrides) -> QLearningAgent:
        values = {"epsilon_start": 0.0, "epsilon_min": 0.0}
        values.update(overrides)
        config = AgentConfig(**values)
        return QLearningAgent(config, seed=7)

    def test_greedy_action_uses_q_values(self) -> None:
        agent = self.make_agent()
        state = (0, 0, 0, 0, 1, 0)
        self.assertEqual(agent.act(state, explore=False), 0)
        agent.q_table[state + (1,)] = 3.5
        self.assertEqual(agent.act(state, explore=False), 1)
        self.assertEqual(agent.best_action(state), 1)

    def test_q_update_bellman(self) -> None:
        agent = self.make_agent(learning_rate=0.5, discount=0.9)
        state = (0, 0, 0, 0, 1, 0)
        next_state = (1, 0, 0, 1, 1, 1)
        agent.q_table[next_state + (0,)] = 10.0
        agent.update(state, 0, 1.0, next_state, done=False)
        self.assertAlmostEqual(agent.q_table[state + (0,)], 0.5 * (1.0 + 0.9 * 10.0))

    def test_time_aware_discount(self) -> None:
        agent = self.make_agent(learning_rate=1.0, discount=0.9)
        state = (0, 0, 0, 0, 1, 0)
        next_state = (1, 0, 0, 1, 1, 1)
        agent.q_table[next_state + (0,)] = 10.0
        agent.update(state, 1, 0.0, next_state, done=False, elapsed=10.0)
        self.assertAlmostEqual(agent.q_table[state + (1,)], 0.9**10 * 10.0)

    def test_terminal_update_has_no_bootstrap(self) -> None:
        agent = self.make_agent(learning_rate=1.0, discount=0.9)
        state = (0, 0, 0, 0, 1, 0)
        next_state = (1, 0, 0, 1, 1, 1)
        agent.q_table[next_state + (0,)] = 10.0
        agent.update(state, 0, 2.0, next_state, done=True)
        self.assertAlmostEqual(agent.q_table[state + (0,)], 2.0)

    def test_invalid_action_rejected(self) -> None:
        agent = self.make_agent()
        with self.assertRaises(ValueError):
            agent.update((0, 0, 0, 0, 1, 0), 9, 0.0, (0, 0, 0, 0, 1, 0))

    def test_epsilon_decay_respects_floor(self) -> None:
        config = AgentConfig(epsilon_start=0.5, epsilon_min=0.1, epsilon_decay=0.5)
        agent = QLearningAgent(config)
        agent.end_episode()
        self.assertAlmostEqual(agent.epsilon, 0.25)
        agent.end_episode()
        self.assertAlmostEqual(agent.epsilon, 0.125)
        agent.end_episode()
        self.assertAlmostEqual(agent.epsilon, 0.1)
        self.assertAlmostEqual(agent.epsilon, 0.1)
        self.assertEqual(agent.episodes_trained, 3)

    def test_seed_makes_exploration_reproducible(self) -> None:
        config = AgentConfig(epsilon_start=1.0)
        a = QLearningAgent(config, seed=42)
        b = QLearningAgent(config, seed=42)
        state = (0, 0, 0, 0, 1, 0)
        actions_a = [a.act(state) for _ in range(100)]
        actions_b = [b.act(state) for _ in range(100)]
        self.assertEqual(actions_a, actions_b)

    def test_save_and_load_roundtrip(self) -> None:
        agent = self.make_agent(epsilon_start=0.4, epsilon_min=0.05)
        state = (2, 1, 3, 0, 1, 2)
        agent.q_table[state + (0,)] = 7.5
        agent.q_table[state + (1,)] = -2.0
        agent.episodes_trained = 12
        agent.total_decisions = 34
        agent.total_updates = 56
        agent.epsilon = 0.123
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.npz"
            saved = agent.save(path)
            self.assertTrue(path.exists())
            self.assertEqual(saved, path)
            loaded = QLearningAgent.load(path, agent.config)
        np.testing.assert_allclose(loaded.q_table, agent.q_table)
        self.assertEqual(loaded.episodes_trained, 12)
        self.assertEqual(loaded.total_decisions, 34)
        self.assertEqual(loaded.total_updates, 56)
        self.assertAlmostEqual(loaded.epsilon, 0.123)

    def test_save_creates_directories(self) -> None:
        agent = self.make_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.npz"
            agent.save(path)
            self.assertTrue(path.exists())

    def test_load_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            QLearningAgent.load("/tmp/autopilot-does-not-exist.npz")

    def test_load_incompatible_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            metadata = json.dumps({"episodes_trained": 1})
            np.savez_compressed(
                path,
                q_table=np.zeros((2, 2)),
                metadata=np.asarray(metadata),
            )
            with self.assertRaises(ValueError):
                QLearningAgent.load(path)

    def test_load_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            path.write_bytes(b"not a numpy file")
            with self.assertRaises(ValueError):
                QLearningAgent.load(path)

    def test_state_shape_has_1024_effective_combinations(self) -> None:
        agent = self.make_agent()
        self.assertEqual(agent.q_table.shape, (4, 4, 4, 2, 2, 4, 2))
        self.assertEqual(agent.q_table.size, 1024 * 2)

    def test_reset_table(self) -> None:
        agent = self.make_agent()
        agent.q_table[0, 0, 0, 0, 1, 0, 0] = 5.0
        agent.episodes_trained = 3
        agent.reset_table()
        self.assertTrue(np.all(agent.q_table == 0.0))
        self.assertEqual(agent.episodes_trained, 0)


if __name__ == "__main__":
    unittest.main()
