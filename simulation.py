"""Simulação física do cruzamento com passo de tempo fixo.

O módulo não conhece Pygame nem Q-Learning. Ele apenas avança veículos,
semáforos e métricas. O ambiente de RL e a interface gráfica são camadas
separadas que consomem esta simulação.
"""

from __future__ import annotations

import math

import numpy as np

from .config import AXIS_OF, DIRECTIONS, Config, SimulationConfig
from .traffic_light import TrafficLight
from .vehicle import DIRECTION_COLORS, Vehicle


class Simulation:
    """Cruzamento com quatro aproximações e faixas independentes."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        seed: int = 0,
        mode: str = "traditional",
    ) -> None:
        self.config = config or Config()
        self.sim_cfg: SimulationConfig = self.config.simulation
        self.vehicles: list[Vehicle] = []
        self.by_direction: dict[str, list[Vehicle]] = {d: [] for d in DIRECTIONS}
        self.reset(seed=seed, mode=mode)

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0, mode: str = "traditional", start_axis: str = "NS") -> None:
        if mode not in ("traditional", "ai"):
            raise ValueError("mode deve ser 'traditional' ou 'ai'")
        self.seed = int(seed)
        self.mode = mode
        self.rng = np.random.default_rng(self.seed)
        self.time = 0.0
        self.vehicles = []
        self.by_direction = {d: [] for d in DIRECTIONS}
        self._next_uid = 0

        # Métricas acumuladas.
        self.total_arrived = 0
        self.total_spawned = 0
        self.total_crossed = 0
        self.total_passed = 0
        self.passed_wait_sum = 0.0
        self.total_wait_seconds = 0.0
        self.max_wait_observed = 0.0
        self.queue_accum = 0.0
        self.queue_samples = 0
        self.queue_ns = 0
        self.queue_ew = 0

        self.light = TrafficLight(
            self.config.lights,
            start_axis=start_axis,
            auto=(mode == "traditional"),
        )

        # Demanda assimétrica por episódio. Os mesmos sorteios são feitos em
        # ambos os modos para que a mesma seed produza as mesmas chegadas.
        ns_factor = self.rng.uniform(*self.sim_cfg.demand_factor_range)
        ew_factor = self.rng.uniform(*self.sim_cfg.demand_factor_range)
        self.demand_factors: dict[str, float] = {}
        self._mean_interval: dict[str, float] = {}
        self._next_arrival: dict[str, float] = {}
        self._pending: dict[str, int] = {d: 0 for d in DIRECTIONS}

        for direction in DIRECTIONS:
            axis_factor = ns_factor if AXIS_OF[direction] == "NS" else ew_factor
            variation = self.rng.uniform(*self.sim_cfg.demand_variation_range)
            factor = max(0.2, axis_factor * variation)
            mean_interval = self.sim_cfg.base_arrival_interval / factor
            self.demand_factors[direction] = factor
            self._mean_interval[direction] = mean_interval
            self._next_arrival[direction] = float(
                self.rng.exponential(mean_interval)
            )

        self._update_queue_metrics()

    # ------------------------------------------------------------------
    # Avanço da simulação
    # ------------------------------------------------------------------
    def step(self, dt: float | None = None) -> None:
        """Avança a simulação em exatamente ``dt`` segundos (passo fixo)."""
        dt = self.sim_cfg.sim_dt if dt is None else float(dt)
        if dt <= 0:
            raise ValueError("dt deve ser positivo")

        self.time += dt
        self.light.update(dt, intersection_clear=not self.intersection_occupied())

        self._enqueue_arrivals()
        self._spawn_pending()

        for direction in DIRECTIONS:
            lane = self.by_direction[direction]
            if not lane:
                continue
            lane.sort(key=lambda v: v.s, reverse=True)
            self._update_lane(direction, lane, dt)

        self._remove_finished()
        self._update_queue_metrics()

    def _enqueue_arrivals(self) -> None:
        """Registra chegadas de acordo com o processo de Poisson."""
        for direction in DIRECTIONS:
            # Enquanto o tempo simulado avança, pode haver mais de uma
            # chegada. O RNG é consumido apenas aqui, independentemente do
            # estado dos semáforos — garantindo comparabilidade entre modos.
            while self.time + 1e-9 >= self._next_arrival[direction]:
                if self._pending[direction] < self.sim_cfg.max_pending_per_direction:
                    self._pending[direction] += 1
                    self.total_arrived += 1
                interval = max(
                    self.sim_cfg.min_arrival_interval,
                    float(self.rng.exponential(self._mean_interval[direction])),
                )
                self._next_arrival[direction] += interval

    def _spawn_pending(self) -> None:
        for direction in DIRECTIONS:
            if self._pending[direction] <= 0:
                continue
            if len(self.vehicles) >= self.sim_cfg.max_vehicles:
                return
            lane = self.by_direction[direction]
            if lane and min(v.s for v in lane) < self.sim_cfg.spawn_clearance:
                continue
            self.spawn_vehicle(direction)
            self._pending[direction] -= 1

    def spawn_vehicle(
        self,
        direction: str,
        *,
        s: float = 0.0,
        speed: float | None = None,
        count_as_arrival: bool = False,
    ) -> Vehicle:
        """Cria um veículo. Também é usado pelos testes."""
        if direction not in DIRECTIONS:
            raise ValueError(f"direção inválida: {direction}")
        if len(self.vehicles) >= self.sim_cfg.max_vehicles:
            raise RuntimeError("limite de veículos atingido")

        uid = self._next_uid
        self._next_uid += 1
        fraction = ((uid * 2654435761) & 0xFFFF) / 65535.0
        desired = self.sim_cfg.max_speed * (0.88 + 0.12 * fraction)
        base_color = DIRECTION_COLORS[direction]
        shade = 0.86 + 0.24 * fraction
        color = tuple(min(255, int(c * shade)) for c in base_color)
        initial_speed = desired * 0.72 if speed is None else float(speed)

        vehicle = Vehicle(
            uid=uid,
            direction=direction,
            s=float(s),
            speed=max(0.0, initial_speed),
            desired_speed=desired,
            length=self.sim_cfg.vehicle_length,
            width=self.sim_cfg.vehicle_width,
            prev_s=float(s),
            color=color,  # type: ignore[arg-type]
        )
        self.vehicles.append(vehicle)
        self.by_direction[direction].append(vehicle)
        if count_as_arrival:
            self.total_arrived += 1
        self.total_spawned += 1
        return vehicle

    # ------------------------------------------------------------------
    # Física
    # ------------------------------------------------------------------
    def _update_lane(self, direction: str, lane: list[Vehicle], dt: float) -> None:
        cfg = self.sim_cfg
        signal = self.light.signal(AXIS_OF[direction])
        leader: Vehicle | None = None

        for vehicle in lane:
            vehicle.prev_s = vehicle.s
            accel = self._free_accel(vehicle)

            if leader is not None:
                accel = min(accel, self._following_accel(vehicle, leader))

            must_stop = self._must_stop_at_light(vehicle, signal)

            if must_stop:
                accel = min(accel, self._stop_line_accel(vehicle))

            accel = max(-cfg.max_brake, min(cfg.max_accel, accel))
            vehicle.speed = max(0.0, min(vehicle.desired_speed, vehicle.speed + accel * dt))
            new_s = vehicle.s + vehicle.speed * dt

            if must_stop and not vehicle.entered and vehicle.s <= cfg.stop_line:
                limit = cfg.stop_line - 0.20
                if new_s > limit:
                    new_s = max(vehicle.s, limit)
                    vehicle.speed = 0.0

            vehicle.s = new_s
            leader = vehicle

        # Garantia dura de não sobreposição, independente do modelo de
        # aceleração. Pequena folga evita toque entre para-choques.
        for index in range(1, len(lane)):
            leader = lane[index - 1]
            follower = lane[index]
            max_s = leader.s - leader.length - 0.30
            if follower.s > max_s:
                follower.s = max(0.0, max_s)
                follower.speed = min(follower.speed, leader.speed)
                if follower.speed < 0.0:
                    follower.speed = 0.0

        for vehicle in lane:
            if not vehicle.entered and vehicle.s >= cfg.stop_line:
                vehicle.entered = True
                self.total_crossed += 1
            if not vehicle.entered and vehicle.speed < 0.5:
                vehicle.wait_time += dt
                self.total_wait_seconds += dt

    def _free_accel(self, vehicle: Vehicle) -> float:
        """Aceleração para alcançar a velocidade desejada em via livre."""
        ratio = vehicle.speed / max(0.1, vehicle.desired_speed)
        return self.sim_cfg.max_accel * (1.0 - ratio**4)

    def _following_accel(self, vehicle: Vehicle, leader: Vehicle) -> float:
        cfg = self.sim_cfg
        gap = leader.s - leader.length - vehicle.s
        if gap <= 0.05:
            return -cfg.max_brake

        dv = vehicle.speed - leader.speed
        sqrt_ab = math.sqrt(cfg.max_accel * cfg.max_brake)
        s_star = cfg.min_gap + max(
            0.0,
            vehicle.speed * cfg.time_headway + vehicle.speed * dv / (2.0 * sqrt_ab),
        )
        ratio = vehicle.speed / max(0.1, vehicle.desired_speed)
        return cfg.max_accel * (1.0 - ratio**4 - (s_star / gap) ** 2)

    def _stop_line_accel(self, vehicle: Vehicle) -> float:
        """IDM com um líder virtual parado na linha de retenção."""
        cfg = self.sim_cfg
        gap = max(0.05, cfg.stop_line - vehicle.s - 0.30)
        sqrt_ab = math.sqrt(cfg.max_accel * cfg.max_brake)
        s_star = cfg.min_gap * 0.6 + vehicle.speed * 0.7 + vehicle.speed**2 / (2.0 * sqrt_ab)
        ratio = vehicle.speed / max(0.1, vehicle.desired_speed)
        return cfg.max_accel * (1.0 - ratio**4 - (s_star / gap) ** 2)

    def _must_stop_at_light(self, vehicle: Vehicle, signal: str) -> bool:
        if vehicle.entered:
            return False
        # Um veículo já posicionado além da linha de retenção não deve
        # frear para um semáforo que ficou para trás (útil em testes e ao
        # restaurar estados).
        if vehicle.s >= self.sim_cfg.stop_line:
            return False
        if signal == "green":
            return False
        if signal == "yellow":
            if vehicle.committed_yellow:
                return False
            gap = self.sim_cfg.stop_line - vehicle.s
            stopping_distance = (
                vehicle.speed**2 / (2.0 * self.sim_cfg.max_brake)
                + vehicle.speed * 0.35
                + 1.0
            )
            if gap <= stopping_distance:
                vehicle.committed_yellow = True
                return False
            return True
        # Vermelho: apenas quem já se comprometeu no amarelo pode concluir.
        return not vehicle.committed_yellow

    # ------------------------------------------------------------------
    # Consultas e métricas
    # ------------------------------------------------------------------
    def intersection_occupied(self) -> bool:
        """True se algum veículo ainda ocupa a caixa do cruzamento."""
        stop = self.sim_cfg.stop_line
        end = stop + self.sim_cfg.intersection_length + 0.5
        for vehicle in self.vehicles:
            if vehicle.s > stop - 0.01 and vehicle.s - vehicle.length < end:
                return True
        return False

    def _remove_finished(self) -> None:
        limit = self.sim_cfg.lane_length
        for direction in DIRECTIONS:
            lane = self.by_direction[direction]
            for vehicle in lane:
                if vehicle.s >= limit and not vehicle.finished:
                    vehicle.finished = True
                    self.total_passed += 1
                    self.passed_wait_sum += vehicle.wait_time
                    self.max_wait_observed = max(self.max_wait_observed, vehicle.wait_time)
            self.by_direction[direction] = [v for v in lane if not v.finished]
        self.vehicles = [v for v in self.vehicles if not v.finished]

    def _update_queue_metrics(self) -> None:
        queue_ns = 0
        queue_ew = 0
        for vehicle in self.vehicles:
            self.max_wait_observed = max(self.max_wait_observed, vehicle.wait_time)
            if vehicle.is_waiting():
                if vehicle.axis == "NS":
                    queue_ns += 1
                else:
                    queue_ew += 1
        self.queue_ns = queue_ns
        self.queue_ew = queue_ew
        total = queue_ns + queue_ew
        self.queue_accum += total
        self.queue_samples += 1

    def metrics_snapshot(self) -> dict[str, float]:
        """Contadores usados para calcular a recompensa por intervalo."""
        return {
            "crossed": float(self.total_crossed),
            "wait_seconds": float(self.total_wait_seconds),
            "queue_accum": float(self.queue_accum),
            "queue_samples": float(self.queue_samples),
            "last_queue": float(self.queue_ns + self.queue_ew),
        }

    @property
    def avg_queue(self) -> float:
        return self.queue_accum / self.queue_samples if self.queue_samples else 0.0

    @property
    def active_count(self) -> int:
        return len(self.vehicles)

    @property
    def avg_wait(self) -> float:
        active_wait = sum(v.wait_time for v in self.vehicles)
        denominator = self.total_spawned
        if denominator <= 0:
            return 0.0
        return (self.passed_wait_sum + active_wait) / denominator

    @property
    def current_avg_wait(self) -> float:
        waiting = [v.wait_time for v in self.vehicles if v.is_waiting()]
        return sum(waiting) / len(waiting) if waiting else 0.0

    def stats(self) -> dict[str, float | int | str]:
        return {
            "time": self.time,
            "seed": self.seed,
            "mode": self.mode,
            "active": self.active_count,
            "arrived": self.total_arrived,
            "spawned": self.total_spawned,
            "crossed": self.total_crossed,
            "passed": self.total_passed,
            "avg_wait": self.avg_wait,
            "current_avg_wait": self.current_avg_wait,
            "max_wait": self.max_wait_observed,
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "avg_queue": self.avg_queue,
            "pending": sum(self._pending.values()),
            "phase": self.light.phase.value,
            "signal_ns": self.light.signal("NS"),
            "signal_ew": self.light.signal("EW"),
        }
