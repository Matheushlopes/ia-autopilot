"""Agente Q-Learning tabular.

O agente é deliberadamente simples e auditável: a Q-table é um array NumPy
com uma entrada por estado e ação. Nenhuma decisão é mascarada como
"aprendizado": quando o agente não foi treinado, a Q-table é zero e a
política gananciosa simplesmente mantém a fase atual.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AgentConfig, Config


State = tuple[int, int, int, int, int, int]

KEEP = 0
SWITCH = 1
ACTION_NAMES = {KEEP: "MANTER", SWITCH: "TROCAR"}


class QLearningAgent:
    """Q-Learning com exploração epsilon-greedy e decaimento por episódio."""

    STATE_SHAPE = (4, 4, 4, 2, 2, 4)

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        seed: int | None = None,
        epsilon: float | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        expected = (
            self.config.queue_bins,
            self.config.queue_bins,
            self.config.wait_bins,
            self.config.phase_bins,
            self.config.min_green_bins,
            self.config.green_age_bins,
        )
        if expected != self.STATE_SHAPE:
            raise ValueError(
                "A configuração de bins não corresponde ao formato de estado "
                f"esperado {self.STATE_SHAPE}: {expected}"
            )
        self.q_table = np.zeros(expected + (self.config.n_actions,), dtype=np.float64)
        self.epsilon = (
            self.config.epsilon_start if epsilon is None else float(epsilon)
        )
        self.episodes_trained = 0
        self.total_decisions = 0
        self.total_updates = 0
        self.rng = np.random.default_rng(seed)
        self.updated_at: str | None = None

    # ------------------------------------------------------------------
    # Política
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.episodes_trained > 0

    def q_values(self, state: Sequence[int]) -> np.ndarray:
        return self.q_table[tuple(int(x) for x in state)]

    def best_action(self, state: Sequence[int]) -> int:
        return int(np.argmax(self.q_values(state)))

    def act(self, state: Sequence[int], *, explore: bool = True) -> int:
        """Escolhe uma ação. ``explore=False`` é usado em avaliação/renderização."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.config.n_actions))
        return self.best_action(state)

    # ------------------------------------------------------------------
    # Aprendizado
    # ------------------------------------------------------------------
    def update(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int],
        *,
        done: bool = False,
        elapsed: float | None = None,
    ) -> float:
        """Aplica a atualização de Bellman e retorna o erro TD.

        ``elapsed`` é o tempo simulado entre as duas decisões. O desconto é
        aplicado como ``gamma ** elapsed``, o que mantém a comparação justa
        entre intervalos de durações diferentes (por exemplo, um intervalo
        que inclui amarelo e vermelho geral).
        """
        s = tuple(int(x) for x in state)
        ns = tuple(int(x) for x in next_state)
        action = int(action)
        if action < 0 or action >= self.config.n_actions:
            raise ValueError(f"ação inválida: {action}")

        current = self.q_table[s + (action,)]
        if done:
            target = float(reward)
        else:
            if elapsed is None:
                discount = self.config.discount
            else:
                discount = self.config.discount ** max(0.0, float(elapsed))
            target = float(reward) + discount * float(np.max(self.q_table[ns]))
        td_error = target - float(current)
        self.q_table[s + (action,)] = current + self.config.learning_rate * td_error
        self.total_updates += 1
        return td_error

    def end_episode(self) -> None:
        self.episodes_trained += 1
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def decay_epsilon(self, decay: float) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * float(decay))

    def reset_table(self) -> None:
        self.q_table.fill(0.0)
        self.episodes_trained = 0
        self.total_decisions = 0
        self.total_updates = 0
        self.epsilon = self.config.epsilon_start
        self.updated_at = None

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            "format": "autopilot-qtable",
            "version": 1,
            "state_shape": list(self.STATE_SHAPE),
            "n_actions": self.config.n_actions,
            "episodes_trained": self.episodes_trained,
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "learning_rate": self.config.learning_rate,
            "discount": self.config.discount,
            "updated_at": self.updated_at,
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else Config().model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        metadata = json.dumps(self.metadata(), ensure_ascii=False)
        try:
            with open(temp, "wb") as handle:
                np.savez_compressed(
                    handle,
                    q_table=self.q_table,
                    metadata=np.asarray(metadata),
                )
            os.replace(temp, target)
        except OSError as exc:
            if temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass
            raise OSError(f"não foi possível salvar o agente em {target}: {exc}") from exc
        return target

    @classmethod
    def load(cls, path: str | Path, config: AgentConfig | None = None) -> "QLearningAgent":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(
                f"Q-table não encontrada em {source}. Treine o agente primeiro."
            )
        try:
            with np.load(source, allow_pickle=False) as data:
                if "q_table" not in data:
                    raise ValueError("arquivo não contém 'q_table'")
                q_table = np.array(data["q_table"], dtype=np.float64)
                metadata: dict[str, Any] = {}
                if "metadata" in data:
                    raw = data["metadata"]
                    text = raw.item() if hasattr(raw, "item") else str(raw)
                    if text:
                        metadata = json.loads(str(text))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"agente inválido em {source}: {exc}") from exc

        agent = cls(config)
        expected = agent.q_table.shape
        if q_table.shape != expected:
            raise ValueError(
                f"formato da Q-table incompatível: {q_table.shape} != {expected}"
            )
        agent.q_table = q_table
        agent.episodes_trained = int(metadata.get("episodes_trained", 0))
        agent.total_decisions = int(metadata.get("total_decisions", 0))
        agent.total_updates = int(metadata.get("total_updates", 0))
        agent.epsilon = float(
            metadata.get("epsilon", agent.config.epsilon_min)
        )
        agent.epsilon = min(
            agent.config.epsilon_start,
            max(agent.config.epsilon_min, agent.epsilon),
        )
        agent.updated_at = metadata.get("updated_at")
        return agent

    def summary(self) -> str:
        status = "treinado" if self.is_trained else "NÃO treinado"
        return (
            f"{status} | episódios={self.episodes_trained} | "
            f"decisões={self.total_decisions} | ε={self.epsilon:.3f}"
        )
