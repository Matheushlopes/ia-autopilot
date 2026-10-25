"""Modelo de um veículo e sua geometria no cruzamento."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SimulationConfig

# Cor base por aproximação — ajuda a acompanhar o fluxo na tela.
DIRECTION_COLORS: dict[str, tuple[int, int, int]] = {
    "north": (56, 189, 248),
    "south": (96, 165, 250),
    "east": (251, 146, 60),
    "west": (244, 114, 182),
}

# Vetor unitário do movimento em coordenadas de tela (y cresce para baixo).
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (-1.0, 0.0),
    "west": (1.0, 0.0),
}


@dataclass
class Vehicle:
    """Veículo em uma faixa, com posição medida em metros percorridos.

    ``s`` cresce do spawn (0) em direção à saída (``2 * approach_length``).
    ``entered`` indica que a frente já cruzou a linha de retenção e o veículo
    está comprometido a limpar o cruzamento.
    """

    uid: int
    direction: str
    s: float
    speed: float
    desired_speed: float
    length: float = 4.6
    width: float = 2.0
    entered: bool = False
    committed_yellow: bool = False
    finished: bool = False
    wait_time: float = 0.0
    prev_s: float = 0.0
    color: tuple[int, int, int] = field(default=(200, 200, 200))

    @property
    def axis(self) -> str:
        return "NS" if self.direction in ("north", "south") else "EW"

    @property
    def position(self) -> float:
        """Posição atual (frente do veículo) em metros desde o spawn."""
        return self.s

    def travel_position(self, config: SimulationConfig, s: float | None = None) -> tuple[float, float]:
        """Converte ``s`` em coordenadas de mundo (metros, centro = 0,0)."""
        value = self.s if s is None else s
        distance = config.approach_length
        offset = config.lane_offset
        if self.direction == "north":
            return (-offset, -distance + value)
        if self.direction == "south":
            return (offset, distance - value)
        if self.direction == "east":
            return (distance - value, -offset)
        if self.direction == "west":
            return (-distance + value, offset)
        raise ValueError(f"direção inválida: {self.direction}")

    def heading(self) -> tuple[float, float]:
        return DIRECTION_VECTORS[self.direction]

    def interpolated_position(self, config: SimulationConfig, alpha: float) -> tuple[float, float]:
        """Posição interpolada para renderização suave entre passos fixos."""
        clamped = max(0.0, min(1.0, alpha))
        s = self.prev_s + (self.s - self.prev_s) * clamped
        return self.travel_position(config, s)

    def is_waiting(self, threshold: float = 1.0) -> bool:
        return (not self.entered) and self.speed < threshold
