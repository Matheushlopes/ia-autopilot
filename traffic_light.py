"""Máquina de estados dos semáforos.

A classe :class:`TrafficLight` conhece apenas as regras de segurança do
cruzamento. Ela não decide *quando* trocar: no modo tradicional a troca é
automática por temporização; no modo IA a troca é solicitada pelo ambiente
(que consulta o agente Q-Learning), mas o amarelo, o vermelho geral e a
exigência de esvaziar a caixa do cruzamento continuam sendo aplicados aqui.

Invariantes de segurança:
* nunca existem verdes simultâneos para NS (Norte/Sul) e EW (Leste/Oeste);
* uma fase só recebe verde depois de amarelo + vermelho geral;
* o vermelho geral só termina quando a caixa do cruzamento está livre.
"""

from __future__ import annotations

from enum import Enum

from .config import TrafficLightConfig


class Phase(Enum):
    NS_GREEN = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    ALL_RED_TO_EW = "ALL_RED_TO_EW"
    EW_GREEN = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"
    ALL_RED_TO_NS = "ALL_RED_TO_NS"


_GREEN_PHASES = {Phase.NS_GREEN, Phase.EW_GREEN}
_YELLOW_PHASES = {Phase.NS_YELLOW, Phase.EW_YELLOW}
_ALL_RED_PHASES = {Phase.ALL_RED_TO_EW, Phase.ALL_RED_TO_NS}

_SIGNAL_COLORS = {
    "green": (52, 211, 153),
    "yellow": (251, 191, 36),
    "red": (248, 113, 113),
}


class TrafficLight:
    """Semáforo de duas fases com amarelo e vermelho de segurança."""

    def __init__(
        self,
        config: TrafficLightConfig | None = None,
        *,
        start_axis: str = "NS",
        auto: bool = False,
    ) -> None:
        if start_axis not in ("NS", "EW"):
            raise ValueError("start_axis deve ser 'NS' ou 'EW'")
        self.config = config or TrafficLightConfig()
        self.auto = auto
        self.axis = start_axis
        self.phase = Phase.NS_GREEN if start_axis == "NS" else Phase.EW_GREEN
        self.elapsed = 0.0
        self.green_elapsed = 0.0
        self.switch_count = 0

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    @property
    def is_green(self) -> bool:
        return self.phase in _GREEN_PHASES

    @property
    def is_yellow(self) -> bool:
        return self.phase in _YELLOW_PHASES

    @property
    def is_all_red(self) -> bool:
        return self.phase in _ALL_RED_PHASES

    @property
    def next_axis(self) -> str:
        return "EW" if self.axis == "NS" else "NS"

    @property
    def can_switch(self) -> bool:
        return self.is_green and self.green_elapsed + 1e-9 >= self.config.min_green

    @property
    def phase_axis(self) -> str:
        """Eixo da fase atual (no vermelho geral, o último eixo verde)."""
        return self.axis

    def signal(self, axis: str) -> str:
        """Retorna ``green``, ``yellow`` ou ``red`` para um eixo."""
        if axis not in ("NS", "EW"):
            raise ValueError("axis deve ser 'NS' ou 'EW'")
        if self.phase in _GREEN_PHASES and axis == self.axis:
            return "green"
        if self.phase in _YELLOW_PHASES and axis == self.axis:
            return "yellow"
        return "red"

    def signal_color(self, axis: str) -> tuple[int, int, int]:
        return _SIGNAL_COLORS[self.signal(axis)]

    def phase_label(self) -> str:
        if self.is_green:
            return f"{self._axis_label()} · VERDE"
        if self.is_yellow:
            return f"{self._axis_label()} · AMARELO"
        return "TODOS VERMELHOS"

    def phase_short(self) -> str:
        if self.is_green:
            return "VERDE"
        if self.is_yellow:
            return "AMARELO"
        return "VERMELHO"

    def time_in_phase(self) -> float:
        return self.elapsed

    def _axis_label(self) -> str:
        return "NORTE/SUL" if self.axis == "NS" else "LESTE/OESTE"

    # ------------------------------------------------------------------
    # Transições
    # ------------------------------------------------------------------
    def update(self, dt: float, intersection_clear: bool = True) -> None:
        """Avança a máquina de estados em ``dt`` segundos.

        ``intersection_clear`` é calculado pela simulação e impede que o
        próximo verde comece enquanto houver qualquer veículo na caixa.
        """
        if dt < 0:
            raise ValueError("dt não pode ser negativo")
        self.elapsed += dt

        if self.is_green:
            self.green_elapsed += dt
            if self.auto and self.elapsed + 1e-9 >= self.config.green_duration:
                self._start_yellow()
        elif self.is_yellow:
            if self.elapsed + 1e-9 >= self.config.yellow_duration:
                self._start_all_red()
        elif self.is_all_red:
            if self.elapsed + 1e-9 >= self.config.all_red_duration and intersection_clear:
                self._start_green(self._pending_axis())

    def request_switch(self) -> bool:
        """Solicita o fim do verde atual.

        Retorna ``False`` se a solicitação for ilegal (fora do verde ou antes
        do tempo mínimo de verde). Nesse caso nenhum estado é alterado.
        """
        if not self.can_switch:
            return False
        self._start_yellow()
        return True

    def force_switch(self) -> bool:
        """Troca forçada usada quando o verde máximo é atingido.

        Ainda respeita a existência de um verde ativo; amarelo e vermelho
        geral são sempre cumpridos.
        """
        if not self.is_green:
            return False
        self._start_yellow()
        return True

    def _start_yellow(self) -> None:
        self.phase = Phase.NS_YELLOW if self.axis == "NS" else Phase.EW_YELLOW
        self.elapsed = 0.0
        self.switch_count += 1

    def _start_all_red(self) -> None:
        nxt = self.next_axis
        self.phase = Phase.ALL_RED_TO_EW if nxt == "EW" else Phase.ALL_RED_TO_NS
        self.elapsed = 0.0

    def _start_green(self, axis: str) -> None:
        if axis not in ("NS", "EW"):
            raise ValueError("axis deve ser 'NS' ou 'EW'")
        self.axis = axis
        self.phase = Phase.NS_GREEN if axis == "NS" else Phase.EW_GREEN
        self.elapsed = 0.0
        self.green_elapsed = 0.0

    def _pending_axis(self) -> str:
        if self.phase is Phase.ALL_RED_TO_EW:
            return "EW"
        if self.phase is Phase.ALL_RED_TO_NS:
            return "NS"
        return self.next_axis
