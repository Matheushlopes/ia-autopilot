"""Testes de movimentação, distância segura e passo fixo."""

from __future__ import annotations

import unittest

from autopilot.config import Config
from autopilot.simulation import Simulation


class TestSimulationMovement(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def make_sim(self, *, mode: str = "ai", start_axis: str = "NS") -> Simulation:
        sim = Simulation(self.config, seed=123, mode=mode)
        sim.reset(seed=123, mode=mode, start_axis=start_axis)
        return sim

    def test_vehicle_stops_at_red_light(self) -> None:
        sim = self.make_sim(mode="ai", start_axis="EW")
        vehicle = sim.spawn_vehicle(
            "north",
            s=self.config.simulation.stop_line - 30.0,
            speed=13.0,
        )
        for _ in range(80):
            sim.step()
        self.assertLessEqual(vehicle.s, self.config.simulation.stop_line + 1e-6)
        self.assertFalse(vehicle.entered)
        self.assertLess(vehicle.speed, 0.05)

    def test_vehicle_crosses_on_green(self) -> None:
        sim = self.make_sim(mode="ai", start_axis="NS")
        vehicle = sim.spawn_vehicle(
            "north",
            s=self.config.simulation.stop_line - 20.0,
            speed=10.0,
        )
        for _ in range(50):
            sim.step()
        self.assertTrue(vehicle.entered)
        self.assertGreater(vehicle.s, self.config.simulation.stop_line)
        self.assertEqual(sim.total_crossed, 1)

    def test_no_overlap_in_a_queue(self) -> None:
        sim = self.make_sim(mode="ai", start_axis="EW")
        leader = sim.spawn_vehicle(
            "north",
            s=self.config.simulation.stop_line - 5.0,
            speed=0.0,
        )
        follower = sim.spawn_vehicle(
            "north",
            s=self.config.simulation.stop_line - 30.0,
            speed=13.0,
        )
        for _ in range(100):
            sim.step()
        gap = leader.s - leader.length - follower.s
        self.assertGreaterEqual(gap, 0.0)
        self.assertLess(follower.s, leader.s)

    def test_vehicle_disappears_after_leaving(self) -> None:
        sim = self.make_sim()
        vehicle = sim.spawn_vehicle(
            "east",
            s=self.config.simulation.lane_length - 0.05,
            speed=1.0,
        )
        sim.step()
        self.assertTrue(vehicle.finished)
        self.assertNotIn(vehicle, sim.vehicles)
        self.assertEqual(sim.total_passed, 1)

    def test_fixed_step_is_independent_of_calls(self) -> None:
        sim = self.make_sim()
        for _ in range(100):
            sim.step()
        self.assertAlmostEqual(sim.time, 10.0, places=6)

    def test_same_seed_is_deterministic(self) -> None:
        sim_a = self.make_sim(mode="traditional")
        sim_b = self.make_sim(mode="traditional")
        for _ in range(300):
            sim_a.step()
            sim_b.step()
        stats_a = sim_a.stats()
        stats_b = sim_b.stats()
        for key in ("time", "arrived", "spawned", "crossed", "passed", "avg_wait"):
            self.assertEqual(stats_a[key], stats_b[key], msg=key)
        positions_a = sorted(
            (v.direction, round(v.s, 6), round(v.speed, 6)) for v in sim_a.vehicles
        )
        positions_b = sorted(
            (v.direction, round(v.s, 6), round(v.speed, 6)) for v in sim_b.vehicles
        )
        self.assertEqual(positions_a, positions_b)

    def test_arrival_schedule_is_identical_across_modes(self) -> None:
        traditional = self.make_sim(mode="traditional")
        ai = self.make_sim(mode="ai")
        for _ in range(600):
            traditional.step()
            ai.step()
        self.assertEqual(traditional.total_arrived, ai.total_arrived)
        for direction in ("north", "south", "east", "west"):
            self.assertAlmostEqual(
                traditional._next_arrival[direction],
                ai._next_arrival[direction],
                places=9,
            )
            self.assertAlmostEqual(
                traditional.demand_factors[direction],
                ai.demand_factors[direction],
                places=9,
            )

    def test_long_run_has_no_conflicting_greens_or_overlaps(self) -> None:
        sim = self.make_sim(mode="traditional")
        for _ in range(2400):
            sim.step()
            ns = sim.light.signal("NS")
            ew = sim.light.signal("EW")
            self.assertFalse(ns == "green" and ew == "green")
            for lane in sim.by_direction.values():
                ordered = sorted(lane, key=lambda v: v.s)
                for behind, ahead in zip(ordered, ordered[1:]):
                    gap = ahead.s - behind.s - behind.length
                    self.assertGreaterEqual(gap, 0.0)
        self.assertGreater(sim.total_passed, 0)

    def test_intersection_occupancy(self) -> None:
        sim = self.make_sim()
        self.assertFalse(sim.intersection_occupied())
        vehicle = sim.spawn_vehicle(
            "north",
            s=self.config.simulation.stop_line + 2.0,
            speed=8.0,
        )
        vehicle.entered = True
        self.assertTrue(sim.intersection_occupied())
        vehicle.s = self.config.simulation.lane_length
        self.assertFalse(sim.intersection_occupied())

    def test_invalid_inputs(self) -> None:
        sim = self.make_sim()
        with self.assertRaises(ValueError):
            sim.step(0.0)
        with self.assertRaises(ValueError):
            sim.spawn_vehicle("diagonal")
        with self.assertRaises(ValueError):
            sim.reset(seed=1, mode="magic")


if __name__ == "__main__":
    unittest.main()
