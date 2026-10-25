"""Testes do treinamento acelerado e da persistência automática."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from autopilot.agent import QLearningAgent
from autopilot.config import Config
from autopilot.training import train


class TestTraining(unittest.TestCase):
    def make_config(self, directory: str) -> Config:
        config = Config()
        root = Path(directory)
        config.model_path = root / "q_table.npz"
        config.chart_path = root / "evaluation.png"
        config.report_path = root / "evaluation.json"
        config.checkpoint_every = 1
        return config

    def test_short_training_saves_a_loadable_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory)
            result = train(
                config,
                episodes=2,
                seconds=15.0,
                seed=5,
                verbose=False,
            )
            self.assertEqual(result["episodes_completed"], 2)
            self.assertFalse(result["stopped"])
            self.assertEqual(len(result["history"]), 2)
            self.assertTrue(config.model_path.exists())

            loaded = QLearningAgent.load(config.model_path, config.agent)
            self.assertEqual(loaded.episodes_trained, 2)
            self.assertGreater(loaded.total_updates, 0)
            self.assertLess(loaded.epsilon, config.agent.epsilon_start)

    def test_stop_event_saves_partial_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory)
            stop_event = threading.Event()
            stop_event.set()
            result = train(
                config,
                episodes=10,
                seconds=10.0,
                seed=6,
                stop_event=stop_event,
                verbose=False,
            )
            self.assertTrue(result["stopped"])
            self.assertEqual(result["episodes_completed"], 0)
            self.assertTrue(config.model_path.exists())
            self.assertTrue(Path(result["model_path"]).exists())


if __name__ == "__main__":
    unittest.main()
