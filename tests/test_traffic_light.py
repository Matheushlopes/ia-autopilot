"""Testes das regras de segurança dos semáforos."""

from __future__ import annotations

import unittest

from autopilot.config import TrafficLightConfig
from autopilot.traffic_light import Phase, TrafficLight


class TestTrafficLightCycle(unittest.TestCase):
    def make_light(self, **overrides) -> TrafficLight:
        config = TrafficLightConfig(**overrides)
        return TrafficLight(config, auto=True)

    def test_traditional_cycle_timing(self) -> None:
        light = self.make_light(green_duration=20.0, yellow_duration=3.0, all_red_duration=1.0)
        self.assertTrue(light.is_green)
        self.assertEqual(light.signal("NS"), "green")
        self.assertEqual(light.signal("EW"), "red")

        light.update(19.9)
        self.assertIs(light.phase, Phase.NS_GREEN)
        light.update(0.1)
        self.assertIs(light.phase, Phase.NS_YELLOW)
        self.assertEqual(light.signal("NS"), "yellow")

        light.update(3.0)
        self.assertIs(light.phase, Phase.ALL_RED_TO_EW)
        self.assertEqual(light.signal("NS"), "red")
        self.assertEqual(light.signal("EW"), "red")

        light.update(1.0, intersection_clear=True)
        self.assertIs(light.phase, Phase.EW_GREEN)
        self.assertEqual(light.signal("EW"), "green")
        self.assertEqual(light.signal("NS"), "red")

    def test_never_two_conflicting_greens(self) -> None:
        light = self.make_light(green_duration=5.0, yellow_duration=1.0, all_red_duration=0.5)
        for _ in range(2000):
            light.update(0.1, intersection_clear=True)
            ns = light.signal("NS")
            ew = light.signal("EW")
            self.assertFalse(ns == "green" and ew == "green")
            self.assertIn(ns, ("green", "yellow", "red"))
            self.assertIn(ew, ("green", "yellow", "red"))

    def test_all_red_waits_until_intersection_is_clear(self) -> None:
        light = self.make_light(green_duration=1.0, yellow_duration=1.0, all_red_duration=0.5)
        light.update(1.0)  # inicia amarelo
        light.update(1.0)  # inicia vermelho geral
        self.assertTrue(light.is_all_red)

        light.update(0.5, intersection_clear=False)
        self.assertTrue(light.is_all_red)
        self.assertEqual(light.signal("EW"), "red")

        light.update(0.1, intersection_clear=True)
        self.assertTrue(light.is_green)
        self.assertEqual(light.axis, "EW")

    def test_switch_respects_minimum_green(self) -> None:
        config = TrafficLightConfig(min_green=8.0, yellow_duration=3.0, all_red_duration=1.0)
        light = TrafficLight(config, start_axis="NS", auto=False)

        self.assertFalse(light.request_switch())
        light.update(7.9)
        self.assertFalse(light.request_switch())
        self.assertTrue(light.is_green)

        light.update(0.1)
        self.assertTrue(light.request_switch())
        self.assertTrue(light.is_yellow)
        self.assertEqual(light.axis, "NS")

    def test_switch_ignored_outside_green(self) -> None:
        config = TrafficLightConfig(min_green=1.0, yellow_duration=2.0, all_red_duration=1.0)
        light = TrafficLight(config, start_axis="NS", auto=False)
        light.update(1.0)
        self.assertTrue(light.request_switch())
        self.assertTrue(light.is_yellow)
        self.assertFalse(light.request_switch())

        light.update(2.0)
        self.assertTrue(light.is_all_red)
        self.assertFalse(light.request_switch())

    def test_force_switch_only_when_green(self) -> None:
        light = TrafficLight(TrafficLightConfig(), auto=False)
        self.assertTrue(light.force_switch())
        self.assertTrue(light.is_yellow)
        self.assertFalse(light.force_switch())

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            TrafficLight(start_axis="XX")
        light = TrafficLight()
        with self.assertRaises(ValueError):
            light.signal("XX")
        with self.assertRaises(ValueError):
            light.update(-0.1)

    def test_full_cycle_returns_to_same_axis(self) -> None:
        light = self.make_light(green_duration=5.0, yellow_duration=1.0, all_red_duration=0.5)
        seen = []
        for _ in range(1000):
            light.update(0.1, intersection_clear=True)
            if light.is_green and light.axis not in seen:
                seen.append(light.axis)
        self.assertEqual(seen[:2], ["NS", "EW"])


if __name__ == "__main__":
    unittest.main()
