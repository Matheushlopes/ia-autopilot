"""Ambiente de RL: observação, recompensa e integração com a simulação.

O ambiente também é usado pela interface gráfica no modo IA. Assim, a mesma
lógica que treina o agente é a lógica que aparece na tela: não existe um
"modo demonstração" com decisões falsas.
"""

from __future__ import annotations

from typing import Any, Sequence

from .agent import KEEP, SWITCH, QLearningAgent
from .config import Config, RewardConfig
from .simulation import Simulation


# ----------------------------------------------------------------------
# Discretização
# ----------------------------------------------------------------------
def bin_value(value: float, thresholds: Sequence[float]) -> int:
    """Retorna o índice da faixa: 0, 1, 2 ou 3 para três limiares."""
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)


def observation_from_simulation(
    sim: Simulation, config: Config
) -> tuple[int, int, int, int, int, int]:
    """Converte o estado contínuo em uma tupla discreta de 6 dimensões."""
    ns_count = 0
    ew_count = 0
    ns_wait = 0.0
    ew_wait = 0.0
    for vehicle in sim.vehicles:
        if not vehicle.is_waiting():
            continue
        if vehicle.axis == "NS":
            ns_count += 1
            ns_wait += vehicle.wait_time
        else:
            ew_count += 1
            ew_wait += vehicle.wait_time

    avg_wait_ns = ns_wait / ns_count if ns_count else 0.0
    avg_wait_ew = ew_wait / ew_count if ew_count else 0.0
    avg_wait = (avg_wait_ns * ns_count + avg_wait_ew * ew_count) / max(
        1, ns_count + ew_count
    )

    queue_ns_bin = bin_value(ns_count, config.agent.queue_thresholds)
    queue_ew_bin = bin_value(ew_count, config.agent.queue_thresholds)
    wait_bin = bin_value(avg_wait, config.agent.wait_thresholds)
    phase = 0 if sim.light.axis == "NS" else 1
    min_green_done = 1 if sim.light.green_elapsed + 1e-9 >= config.lights.min_green else 0
    green_age_bin = bin_value(
        sim.light.green_elapsed, config.agent.green_age_thresholds
    )
    return (
        queue_ns_bin,
        queue_ew_bin,
        wait_bin,
        phase,
        min_green_done,
        green_age_bin,
    )


# ----------------------------------------------------------------------
# Recompensa
# ----------------------------------------------------------------------
def reward_components(
    previous: dict[str, float],
    current: dict[str, float],
    config: RewardConfig,
) -> dict[str, float]:
    """Calcula as parcelas da recompensa a partir de contadores reais."""
    crossed = max(0.0, current["crossed"] - previous["crossed"])
    wait_seconds = max(0.0, current["wait_seconds"] - previous["wait_seconds"])
    queue_samples = current["queue_samples"] - previous["queue_samples"]
    if queue_samples > 0:
        avg_queue = (current["queue_accum"] - previous["queue_accum"]) / queue_samples
    else:
        avg_queue = current["last_queue"]

    throughput = config.throughput_reward * crossed
    wait_penalty = config.waiting_penalty_per_vehicle_second * wait_seconds
    queue_penalty = config.queue_penalty_per_vehicle * avg_queue
    long_queue_penalty = config.long_queue_penalty * max(
        0.0, avg_queue - config.long_queue_threshold
    )
    total = throughput - wait_penalty - queue_penalty - long_queue_penalty
    return {
        "crossed": crossed,
        "wait_seconds": wait_seconds,
        "avg_queue": avg_queue,
        "throughput": throughput,
        "wait_penalty": wait_penalty,
        "queue_penalty": queue_penalty,
        "long_queue_penalty": long_queue_penalty,
        "total": total,
    }


def compute_interval_reward(
    previous: dict[str, float],
    current: dict[str, float],
    config: RewardConfig,
) -> float:
    return reward_components(previous, current, config)["total"]


class TrafficEnvironment:
    """Executa episódios de simulação e aplica a política do agente."""

    def __init__(self, config: Config | None = None, *, agent: QLearningAgent | None = None) -> None:
        self.config = config or Config()
        self.sim = Simulation(self.config)
        self.agent = agent
        self.mode = "traditional"
        self.training = False
        self.episode_seconds = float("inf")
        self.done = False
        self.episode_reward = 0.0
        self.decisions = 0
        self.switches = 0
        self.last_action = KEEP
        self.last_reward = 0.0
        self.seed = 0
        self.episode_index = 0
        self._pending_state: tuple[int, int, int, int, int, int] | None = None
        self._pending_action = KEEP
        self._snapshot: dict[str, float] | None = None
        self._next_decision = 0.0
        self._pending_switch_penalty = 0.0

    # ------------------------------------------------------------------
    # Ciclo de episódio
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int,
        mode: str = "traditional",
        training: bool = False,
        episode_seconds: float | None = None,
        start_axis: str = "NS",
    ) -> tuple[int, int, int, int, int, int] | None:
        if mode not in ("traditional", "ai"):
            raise ValueError("mode deve ser 'traditional' ou 'ai'")
        self.mode = mode
        self.training = bool(training and self.agent is not None)
        self.seed = int(seed)
        self.episode_seconds = (
            float(episode_seconds)
            if episode_seconds is not None
            else self.config.training_seconds
        )
        self.done = False
        self.episode_reward = 0.0
        self.decisions = 0
        self.switches = 0
        self.last_action = KEEP
        self.last_reward = 0.0
        self._pending_switch_penalty = 0.0
        self._last_decision_time = 0.0
        self.sim.reset(seed=self.seed, mode=mode, start_axis=start_axis)

        if mode == "ai":
            self._pending_state = self.observation()
            self._pending_action = KEEP
            self._snapshot = self.sim.metrics_snapshot()
            self._next_decision = self.sim.time + self.config.lights.min_green
            return self._pending_state

        self._pending_state = None
        self._snapshot = None
        return None

    def update(self, dt: float | None = None) -> None:
        """Avança um passo fixo. A decisão da IA ocorre quando é devida."""
        if self.done:
            return
        self.sim.step(dt)

        if self.mode == "ai":
            self._maybe_decide()

        if self.sim.time + 1e-9 >= self.episode_seconds:
            self._finish_episode()

    def run(self) -> None:
        """Executa até o fim do episódio (útil para treino/avaliação)."""
        while not self.done:
            self.update(self.config.simulation.sim_dt)

    def _maybe_decide(self) -> None:
        light = self.sim.light
        if not light.is_green:
            return
        if light.green_elapsed + 1e-9 < self.config.lights.min_green:
            return
        if self.sim.time + 1e-9 < self._next_decision:
            return

        current = self.sim.metrics_snapshot()
        reward = (
            compute_interval_reward(self._snapshot or current, current, self.config.reward)
            + self._pending_switch_penalty
        )
        self._pending_switch_penalty = 0.0
        state = self.observation()

        if self.training and self.agent is not None and self._pending_state is not None:
            self.agent.update(
                self._pending_state,
                self._pending_action,
                reward,
                state,
                done=False,
                elapsed=max(0.0, self.sim.time - self._last_decision_time),
            )
            self.agent.total_decisions += 1
        self._last_decision_time = self.sim.time

        if self.agent is not None:
            action = self.agent.act(state, explore=self.training)
        else:
            action = KEEP

        # Segurança: nenhum verde dura para sempre, mesmo que a política
        # escolha "manter". A troca forçada ainda cumpre amarelo e vermelho.
        if light.green_elapsed + 1e-9 >= self.config.lights.max_green:
            action = SWITCH

        applied = KEEP
        if action == SWITCH and light.request_switch():
            applied = SWITCH
            self.switches += 1
            self._pending_switch_penalty -= self.config.reward.switch_penalty

        self.last_action = applied
        self.last_reward = reward
        self.episode_reward += reward
        self.decisions += 1
        self._pending_state = state
        self._pending_action = applied
        self._snapshot = current
        self._next_decision = self.sim.time + self.config.lights.ai_decision_interval

    def _finish_episode(self) -> None:
        if self.done:
            return
        if self.mode == "ai" and self._pending_state is not None:
            current = self.sim.metrics_snapshot()
            reward = (
                compute_interval_reward(self._snapshot or current, current, self.config.reward)
                + self._pending_switch_penalty
            )
            self._pending_switch_penalty = 0.0
            state = self.observation()
            if self.training and self.agent is not None:
                self.agent.update(
                    self._pending_state,
                    self._pending_action,
                    reward,
                    state,
                    done=True,
                    elapsed=max(0.0, self.sim.time - self._last_decision_time),
                )
                self.agent.total_decisions += 1
            self.episode_reward += reward
            self.last_reward = reward
            self._pending_state = None
            self._snapshot = current
        self.done = True

    # ------------------------------------------------------------------
    # Observações e resultados
    # ------------------------------------------------------------------
    def observation(self) -> tuple[int, int, int, int, int, int]:
        return observation_from_simulation(self.sim, self.config)

    def final_metrics(self) -> dict[str, Any]:
        sim = self.sim
        return {
            "mode": self.mode,
            "seed": self.seed,
            "avg_wait": sim.avg_wait,
            "served": sim.total_passed,
            "crossed": sim.total_crossed,
            "arrived": sim.total_arrived,
            "spawned": sim.total_spawned,
            "avg_queue": sim.avg_queue,
            "max_wait": sim.max_wait_observed,
            "episode_reward": self.episode_reward,
            "decisions": self.decisions,
            "switches": self.switches,
            "duration": sim.time,
        }

    def panel_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = dict(self.sim.stats())
        stats.update(
            {
                "mode": self.mode,
                "episode_reward": self.episode_reward,
                "last_reward": self.last_reward,
                "last_action": self.last_action,
                "decisions": self.decisions,
                "switches": self.switches,
                "training": self.training,
                "done": self.done,
            }
        )
        return stats
