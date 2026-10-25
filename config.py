"""Configurações centralizadas do AutoPilot.

Todos os módulos (simulação, agente, treinamento, avaliação e interface)
importam seus parâmetros daqui. Assim, mudar uma constante não exige
procurar valores espalhados pelo código.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "q_table.npz"
DEFAULT_CHART_PATH = MODEL_DIR / "evaluation.png"
DEFAULT_REPORT_PATH = MODEL_DIR / "evaluation.json"

DIRECTIONS = ("north", "south", "east", "west")
AXIS_OF = {"north": "NS", "south": "NS", "east": "EW", "west": "EW"}
AXES = ("NS", "EW")

DIRECTION_LABELS = {
    "north": "Norte",
    "south": "Sul",
    "east": "Leste",
    "west": "Oeste",
}


@dataclass
class SimulationConfig:
    """Parâmetros físicos e de tráfego (unidades do SI)."""

    # Passo de tempo fixo da física (segundos). A renderização nunca altera
    # este valor: o loop gráfico apenas acumula tempo e chama passos fixos.
    sim_dt: float = 0.1

    # Geometria (metros).
    approach_length: float = 100.0
    road_half_width: float = 8.0
    lane_offset: float = 4.0

    # Veículos.
    vehicle_length: float = 4.6
    vehicle_width: float = 2.0
    max_speed: float = 13.5
    max_accel: float = 2.4
    max_brake: float = 6.0
    min_gap: float = 2.0
    time_headway: float = 1.1
    spawn_clearance: float = 11.0

    # Demanda (processo de Poisson independente por aproximação).
    base_arrival_interval: float = 6.0
    demand_factor_range: tuple[float, float] = (0.70, 1.45)
    demand_variation_range: tuple[float, float] = (0.88, 1.12)
    min_arrival_interval: float = 0.8

    # Limites de segurança / desempenho.
    max_vehicles: int = 160
    max_pending_per_direction: int = 80

    @property
    def stop_line(self) -> float:
        """Distância percorrida desde o spawn até a linha de retenção."""
        return self.approach_length - self.road_half_width

    @property
    def intersection_length(self) -> float:
        """Extensão da caixa do cruzamento ao longo de uma via."""
        return 2.0 * self.road_half_width

    @property
    def lane_length(self) -> float:
        """Distância total percorrida entre o spawn e a saída do mapa."""
        return 2.0 * self.approach_length


@dataclass
class TrafficLightConfig:
    """Tempos dos semáforos e limites de segurança."""

    # Modo tradicional (segundos).
    green_duration: float = 20.0
    yellow_duration: float = 3.0
    all_red_duration: float = 1.0

    # Regras compartilhadas com a IA.
    min_green: float = 8.0
    max_green: float = 40.0
    ai_decision_interval: float = 2.0


@dataclass
class AgentConfig:
    """Hiperparâmetros do Q-Learning tabular."""

    learning_rate: float = 0.15
    discount: float = 0.97
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    n_actions: int = 2

    # Discretização do estado (4 x 4 x 4 x 2 x 2 x 4 = 1024 estados).
    queue_bins: int = 4
    wait_bins: int = 4
    phase_bins: int = 2
    min_green_bins: int = 2
    green_age_bins: int = 4

    # Limiares das faixas (contagens, segundos e tempo de verde).
    queue_thresholds: tuple[int, int, int] = (1, 4, 8)
    wait_thresholds: tuple[float, float, float] = (5.0, 15.0, 30.0)
    green_age_thresholds: tuple[float, float, float] = (10.0, 20.0, 30.0)


@dataclass
class RewardConfig:
    """Recompensas calculadas a partir dos eventos reais da simulação."""

    throughput_reward: float = 1.0
    waiting_penalty_per_vehicle_second: float = 0.02
    queue_penalty_per_vehicle: float = 0.05
    long_queue_threshold: float = 12.0
    long_queue_penalty: float = 0.03
    switch_penalty: float = 2.0


@dataclass
class Config:
    """Agrupa toda a configuração do projeto."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    lights: TrafficLightConfig = field(default_factory=TrafficLightConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Treinamento acelerado.
    training_episodes: int = 600
    training_seconds: float = 240.0
    training_seed: int = 1
    checkpoint_every: int = 50

    # Avaliação comparativa.
    evaluation_episodes: int = 16
    evaluation_seed: int = 10_000

    # Persistência.
    model_path: Path = DEFAULT_MODEL_PATH
    chart_path: Path = DEFAULT_CHART_PATH
    report_path: Path = DEFAULT_REPORT_PATH

    def ensure_dirs(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.chart_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RenderConfig:
    """Aparência e layout da interface gráfica."""

    width: int = 1280
    height: int = 800
    panel_width: int = 420
    fps: int = 60
    scale: float = 3.6
    speed_options: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)

    # Paleta.
    background: tuple[int, int, int] = (13, 18, 27)
    panel_bg: tuple[int, int, int] = (17, 24, 35)
    panel_card: tuple[int, int, int] = (25, 35, 50)
    text: tuple[int, int, int] = (229, 236, 246)
    text_muted: tuple[int, int, int] = (139, 155, 177)
    accent: tuple[int, int, int] = (64, 224, 208)
    accent_blue: tuple[int, int, int] = (96, 165, 250)
    green: tuple[int, int, int] = (52, 211, 153)
    yellow: tuple[int, int, int] = (251, 191, 36)
    red: tuple[int, int, int] = (248, 113, 113)
    road: tuple[int, int, int] = (43, 52, 66)
    road_edge: tuple[int, int, int] = (31, 39, 51)
    sidewalk: tuple[int, int, int] = (55, 66, 82)
    lane_mark: tuple[int, int, int] = (196, 184, 132)
    white: tuple[int, int, int] = (238, 242, 247)

    @property
    def main_width(self) -> int:
        return self.width - self.panel_width

    @property
    def center(self) -> tuple[float, float]:
        return (self.main_width / 2.0, self.height / 2.0)


DEFAULT_RENDER = RenderConfig()
