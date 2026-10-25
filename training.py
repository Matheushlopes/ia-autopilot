"""Treinamento acelerado, sem renderização.

O laço abaixo roda a mesma :class:`TrafficEnvironment` usada na interface,
com o passo fixo da simulação. O Pygame não é importado aqui, então o
treinamento pode rodar em terminal, servidor ou thread de background.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from .agent import QLearningAgent
from .config import Config
from .environment import TrafficEnvironment

ProgressCallback = Callable[[dict[str, Any]], None]


def _format_seconds(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    return f"{minutes:.1f}min"


def train(
    config: Config | None = None,
    *,
    episodes: int | None = None,
    seconds: float | None = None,
    seed: int | None = None,
    agent: QLearningAgent | None = None,
    fresh: bool = True,
    progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Treina o agente por ``episodes`` episódios e salva a Q-table.

    O treinamento pode ser interrompido por ``stop_event`` ou Ctrl+C; em
    ambos os casos o agente é salvo antes de retornar.
    """
    cfg = config or Config()
    cfg.ensure_dirs()
    episodes = int(cfg.training_episodes if episodes is None else episodes)
    seconds = float(cfg.training_seconds if seconds is None else seconds)
    seed = int(cfg.training_seed if seed is None else seed)
    if episodes <= 0:
        raise ValueError("episodes deve ser positivo")
    if seconds <= 0:
        raise ValueError("seconds deve ser positivo")

    if agent is None:
        agent = QLearningAgent(cfg.agent, seed=seed + 900_001)
    elif fresh:
        agent.reset_table()
    if fresh:
        agent.epsilon = cfg.agent.epsilon_start

    env = TrafficEnvironment(cfg, agent=agent)
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    interrupted = False
    stopped = False
    completed = 0

    log_every = max(1, episodes // 30)
    if verbose:
        print("=" * 78)
        print("AutoPilot — treinamento Q-Learning")
        print(
            f"Episódios: {episodes} | duração: {seconds:.0f}s simulados | "
            f"dt: {cfg.simulation.sim_dt:.3f}s | seed: {seed}"
        )
        print(
            f"α={cfg.agent.learning_rate} | γ={cfg.agent.discount} | "
            f"ε inicial={agent.epsilon:.3f} | ε mín={cfg.agent.epsilon_min}"
        )
        print("=" * 78, flush=True)

    try:
        for episode_index in range(episodes):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

            env.reset(
                seed=seed + episode_index,
                mode="ai",
                training=True,
                episode_seconds=seconds,
            )
            env.run()
            agent.end_episode()
            completed += 1

            metrics = env.final_metrics()
            record = {
                "episode": agent.episodes_trained,
                "reward": metrics["episode_reward"],
                "avg_wait": metrics["avg_wait"],
                "served": metrics["served"],
                "avg_queue": metrics["avg_queue"],
                "max_wait": metrics["max_wait"],
                "decisions": metrics["decisions"],
                "switches": metrics["switches"],
                "epsilon": agent.epsilon,
            }
            history.append(record)

            if progress is not None:
                progress(
                    {
                        "episode": episode_index + 1,
                        "episodes": episodes,
                        "completed": completed,
                        "epsilon": agent.epsilon,
                        "reward": record["reward"],
                        "avg_wait": record["avg_wait"],
                        "served": record["served"],
                        "avg_queue": record["avg_queue"],
                        "elapsed": time.monotonic() - started,
                    }
                )

            if verbose and (
                episode_index < 5
                or (episode_index + 1) % log_every == 0
                or episode_index + 1 == episodes
            ):
                recent = history[-20:]
                avg_reward = sum(item["reward"] for item in recent) / len(recent)
                avg_wait = sum(item["avg_wait"] for item in recent) / len(recent)
                served = sum(item["served"] for item in recent) / len(recent)
                elapsed = time.monotonic() - started
                eta = (elapsed / max(1, episode_index + 1)) * (
                    episodes - episode_index - 1
                )
                print(
                    f"ep {episode_index + 1:4d}/{episodes} | "
                    f"R(20)={avg_reward:7.2f} | espera(20)={avg_wait:5.1f}s | "
                    f"atendidos(20)={served:5.1f} | ε={agent.epsilon:.3f} | "
                    f"t={_format_seconds(elapsed)} eta={_format_seconds(eta)}",
                    flush=True,
                )

            if cfg.checkpoint_every > 0 and completed % cfg.checkpoint_every == 0:
                agent.save(cfg.model_path)
                if verbose:
                    print(f"  checkpoint salvo em {cfg.model_path}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\nTreinamento interrompido pelo usuário. Salvando agente...", flush=True)

    model_path = agent.save(cfg.model_path)
    elapsed = time.monotonic() - started

    if verbose:
        status = "interrompido" if (interrupted or stopped) else "concluído"
        print("-" * 78)
        print(
            f"Treinamento {status}: {completed}/{episodes} episódios em "
            f"{_format_seconds(elapsed)}."
        )
        print(f"Agente salvo em: {model_path}")
        print(f"Épsilon final: {agent.epsilon:.4f} | decisões: {agent.total_decisions}")
        print("-" * 78, flush=True)

    return {
        "episodes_completed": completed,
        "episodes_requested": episodes,
        "interrupted": interrupted,
        "stopped": stopped,
        "epsilon": agent.epsilon,
        "history": history,
        "model_path": str(model_path),
        "elapsed": elapsed,
        "agent": agent,
    }


class TrainingJob:
    """Executa :func:`train` em uma thread, com progresso compartilhado.

    A interface gráfica nunca é tocada pela thread; ela apenas lê
    ``progress`` e ``done`` de forma atômica.
    """

    def __init__(
        self,
        config: Config,
        *,
        episodes: int | None = None,
        seconds: float | None = None,
        seed: int | None = None,
        fresh: bool = True,
    ) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self.progress: dict[str, Any] = {}
        self.started = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="autopilot-training",
            daemon=True,
            kwargs={
                "episodes": episodes,
                "seconds": seconds,
                "seed": seed,
                "fresh": fresh,
            },
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self, **kwargs: Any) -> None:
        try:
            self.result = train(
                self.config,
                progress=self._on_progress,
                stop_event=self.stop_event,
                **kwargs,
            )
        except BaseException as exc:  # a interface mostra o erro ao usuário
            self.error = exc

    def _on_progress(self, info: dict[str, Any]) -> None:
        self.progress = dict(info)
