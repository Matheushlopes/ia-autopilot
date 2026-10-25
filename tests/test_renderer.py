"""Teste de fumaça da interface Pygame em modo headless."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from autopilot.agent import QLearningAgent
from autopilot.config import Config
from autopilot.environment import TrafficEnvironment
from autopilot.renderer import Renderer


class TestRendererSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.display.init()
        pygame.display.set_mode((1280, 800))
        pygame.font.init()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.display.quit()

    def setUp(self) -> None:
        self.config = Config()
        self.renderer = Renderer(self.config)
        self.agent = QLearningAgent(self.config.agent, seed=4)
        self.env = TrafficEnvironment(self.config, agent=self.agent)
        self.env.reset(seed=4, mode="ai", training=False, episode_seconds=20.0)
        for _ in range(120):
            self.env.update()

    def test_draw_traditional_and_ai(self) -> None:
        ui = {"running": True, "speed": 1.0, "status": "Pronto"}
        self.renderer.draw(self.env, self.agent, ui)

        traditional = TrafficEnvironment(self.config)
        traditional.reset(seed=4, mode="traditional", training=False, episode_seconds=20.0)
        for _ in range(80):
            traditional.update()
        self.renderer.draw(traditional, None, ui)

    def test_buttons_cover_every_action(self) -> None:
        ui = {"running": True, "speed": 1.0, "status": ""}
        self.renderer.draw(self.env, self.agent, ui)
        actions = {button.action for button in self.renderer.buttons}
        self.assertEqual(
            actions,
            {
                "start",
                "restart",
                "mode",
                "speed_down",
                "speed_up",
                "train",
                "save",
                "load",
                "compare",
                "quit",
            },
        )
        for button in self.renderer.buttons:
            self.assertEqual(
                self.renderer.button_at(button.rect.center),
                button.action,
            )

    def test_draw_pause_training_evaluation_and_comparison_overlays(self) -> None:
        ui = {"running": False, "speed": 2.0, "status": "Pausado"}
        self.renderer.draw(self.env, self.agent, ui)

        ui = {
            "running": True,
            "speed": 4.0,
            "status": "Treinando",
            "training": {
                "episode": 12,
                "episodes": 50,
                "reward": 12.5,
                "avg_wait": 8.4,
                "served": 120,
                "epsilon": 0.42,
            },
        }
        self.renderer.draw(self.env, self.agent, ui)

        ui = {
            "running": True,
            "speed": 1.0,
            "status": "Avaliando",
            "comparison_running": {"episode": 3, "episodes": 10},
        }
        self.renderer.draw(self.env, self.agent, ui)

        fake_result = {
            "episodes": 2,
            "traditional": {
                "avg_wait": 10.0,
                "served_mean": 120.0,
                "avg_queue": 7.5,
                "max_wait": 30.0,
            },
            "ai": {
                "avg_wait": 9.0,
                "served_mean": 121.0,
                "avg_queue": 7.0,
                "max_wait": 28.0,
            },
            "chart_path": "models/evaluation.png",
            "report_path": "models/evaluation.json",
        }
        ui = {
            "running": True,
            "speed": 1.0,
            "status": "Comparação concluída",
            "comparison": fake_result,
        }
        self.renderer.draw(self.env, self.agent, ui)


if __name__ == "__main__":
    unittest.main()
