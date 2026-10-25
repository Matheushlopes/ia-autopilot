"""Ponto de entrada do AutoPilot.

Uso:
    python -m autopilot.main                 # interface gráfica
    python -m autopilot.main --train         # treinamento acelerado
    python -m autopilot.main --evaluate      # comparação tradicional x IA
    python -m autopilot.main --train --evaluate

Também funciona como script: ``python autopilot/main.py``.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# Permite executar o arquivo diretamente (python autopilot/main.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autopilot.agent import QLearningAgent  # noqa: E402
from autopilot.config import Config, DEFAULT_MODEL_PATH  # noqa: E402
from autopilot.environment import TrafficEnvironment  # noqa: E402
from autopilot.evaluation import compare_controllers  # noqa: E402
from autopilot.renderer import Renderer  # noqa: E402
from autopilot.training import TrainingJob, train  # noqa: E402


# ----------------------------------------------------------------------
# Trabalho de avaliação em background
# ----------------------------------------------------------------------
class ComparisonJob:
    def __init__(self, config: Config, agent: QLearningAgent, *, episodes: int, seconds: float) -> None:
        self.config = config
        self.agent = agent
        self.episodes = episodes
        self.seconds = seconds
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self.progress: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name="autopilot-eval", daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.result = compare_controllers(
                self.config,
                agent=self.agent,
                episodes=self.episodes,
                seconds=self.seconds,
                progress=self._on_progress,
            )
        except BaseException as exc:
            self.error = exc

    def _on_progress(self, info: dict[str, Any]) -> None:
        self.progress = dict(info)


# ----------------------------------------------------------------------
# Aplicação gráfica
# ----------------------------------------------------------------------
class App:
    def __init__(self, config: Config, *, seed: int | None = None) -> None:
        import pygame

        self.pygame = pygame
        self.config = config
        from autopilot.config import RenderConfig

        self.rc = RenderConfig()
        pygame.init()
        pygame.display.set_caption("AutoPilot — Semáforos com Q-Learning")
        self.screen = pygame.display.set_mode((self.rc.width, self.rc.height))
        pygame.display.set_icon(self._make_icon())
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(config, self.rc)

        self.agent: QLearningAgent | None = None
        self.status = (
            "Pronto. A simulação usa passo fixo de "
            f"{config.simulation.sim_dt:.2f}s, independente do FPS."
        )
        self._try_autoload()
        self.env = TrafficEnvironment(config, agent=self.agent)
        self.seed = int(seed if seed is not None else random.randint(1, 999_999))
        self.mode = "traditional"
        self.env.reset(
            seed=self.seed,
            mode=self.mode,
            training=False,
            episode_seconds=float("inf"),
        )
        self.running = True
        self.accumulator = 0.0
        self.speed_index = list(self.rc.speed_options).index(1.0)
        self.fps = 60.0
        if self.agent is None:
            self.status += " Pressione T para treinar a IA."
        else:
            self.status = (
                f"Agente carregado ({self.agent.episodes_trained} episódios). "
                "Pressione M para ver a IA controlando."
            )

        self.job: TrainingJob | None = None
        self.comparison_job: ComparisonJob | None = None
        self.comparison: dict[str, Any] | None = None
        self._comparison_pending = False
        self._quit_after_job = False
        self._closing = False
        self._job_finalized = True
        self._comparison_finalized = True
        self._last_tick = time.monotonic()

    # ------------------------------------------------------------------
    def _make_icon(self) -> Any:
        pygame = self.pygame
        icon = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.rect(icon, (17, 24, 35), icon.get_rect(), border_radius=8)
        for index, color in enumerate(((248, 113, 113), (251, 191, 36), (52, 211, 153))):
            pygame.draw.circle(icon, color, (16, 8 + index * 8), 3)
        return icon

    def _try_autoload(self) -> None:
        path = self.config.model_path
        if not path.exists():
            self.agent = None
            return
        try:
            self.agent = QLearningAgent.load(path, self.config.agent)
        except (OSError, ValueError) as exc:
            self.agent = None
            self.status = f"Falha ao carregar agente: {exc}"

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    @property
    def speed(self) -> float:
        return self.rc.speed_options[self.speed_index]

    def _set_status(self, message: str) -> None:
        self.status = message

    def _busy(self) -> bool:
        return bool(self.job and self.job.running) or bool(
            self.comparison_job and self.comparison_job.running
        )

    def toggle_pause(self) -> None:
        if self._busy():
            self._set_status("Aguarde a operação em andamento ou interrompa o treino.")
            return
        self.running = not self.running
        self._set_status("Simulação em andamento." if self.running else "Simulação pausada.")

    def restart(self, *, new_seed: bool = True, keep_mode: bool = True) -> None:
        if self._busy():
            self._set_status("Não é possível reiniciar durante uma operação.")
            return
        if new_seed:
            self.seed = random.randint(1, 999_999)
        if not keep_mode:
            self.mode = "traditional"
        self.env.agent = self.agent
        self.env.reset(
            seed=self.seed,
            mode=self.mode,
            training=False,
            episode_seconds=float("inf"),
        )
        self.accumulator = 0.0
        self.running = True
        self._set_status(f"Sessão reiniciada · seed {self.seed} · modo {self._mode_label()}.")

    def toggle_mode(self) -> None:
        if self._busy():
            self._set_status("Não é possível trocar de modo durante uma operação.")
            return
        self.mode = "ai" if self.mode == "traditional" else "traditional"
        self.env.agent = self.agent
        self.env.reset(
            seed=self.seed,
            mode=self.mode,
            training=False,
            episode_seconds=float("inf"),
        )
        self.accumulator = 0.0
        self.running = True
        if self.mode == "ai" and (self.agent is None or not self.agent.is_trained):
            self._set_status(
                "Modo IA com agente NÃO treinado: a Q-table está vazia e a fase "
                "só muda por segurança (verde máximo). Treine com T."
            )
        else:
            self._set_status(f"Modo {self._mode_label()} · seed {self.seed}.")

    def _mode_label(self) -> str:
        return "IA (Q-Learning)" if self.mode == "ai" else "tradicional"

    def change_speed(self, direction: int) -> None:
        old = self.speed
        self.speed_index = max(0, min(len(self.rc.speed_options) - 1, self.speed_index + direction))
        if self.speed != old:
            self._set_status(f"Velocidade da simulação: {self.speed:.1f}x.")

    def save_agent(self) -> None:
        if self.agent is None or (not self.agent.is_trained and self.agent.total_updates == 0):
            self._set_status("Nada para salvar: treine a IA primeiro.")
            return
        try:
            path = self.agent.save(self.config.model_path)
        except OSError as exc:
            self._set_status(f"Erro ao salvar: {exc}")
            return
        self._set_status(f"Agente salvo em {path}.")

    def load_agent(self) -> None:
        if self._busy():
            self._set_status("Não é possível carregar durante uma operação.")
            return
        if not self.config.model_path.exists():
            self._set_status(
                f"Nenhum agente encontrado em {self.config.model_path}. Treine com T."
            )
            return
        try:
            agent = QLearningAgent.load(self.config.model_path, self.config.agent)
        except (OSError, ValueError) as exc:
            self._set_status(f"Erro ao carregar agente: {exc}")
            return
        self.agent = agent
        self.env.agent = agent
        self.env.reset(
            seed=self.seed,
            mode=self.mode,
            training=False,
            episode_seconds=float("inf"),
        )
        self.accumulator = 0.0
        self.running = True
        self._set_status(
            f"Agente carregado: {agent.episodes_trained} episódios · "
            f"{agent.total_decisions} decisões."
        )

    def toggle_training(self) -> None:
        if self.job is not None and self.job.running:
            self.job.stop()
            self._set_status("Interrompendo treinamento com segurança ao fim do episódio...")
            return
        if self.comparison_job is not None and self.comparison_job.running:
            self._set_status("Aguarde a avaliação terminar.")
            return
        self.env.agent = self.agent
        self.running = False
        self.job = TrainingJob(
            self.config,
            episodes=self.config.training_episodes,
            seconds=self.config.training_seconds,
            seed=self.config.training_seed,
            fresh=True,
        )
        self.job.start()
        self._job_finalized = False
        self._set_status(
            f"Treinando {self.config.training_episodes} episódios em background. "
            "O progresso aparece aqui e no terminal. T interrompe."
        )

    def start_comparison(self) -> None:
        if self._busy():
            self._set_status("Aguarde a operação atual.")
            return
        if self.agent is None or not self.agent.is_trained:
            self._set_status(
                "A comparação exige um agente treinado. Pressione T ou carregue uma Q-table."
            )
            return
        self.running = False
        self.comparison = None
        self.comparison_job = ComparisonJob(
            self.config,
            self.agent,
            episodes=self.config.evaluation_episodes,
            seconds=self.config.training_seconds,
        )
        self.comparison_job.start()
        self._comparison_finalized = False
        self._comparison_pending = True
        self._set_status(
            f"Avaliando {self.config.evaluation_episodes} episódios com as mesmas seeds "
            "(sem exploração, sem atualizar a Q-table)..."
        )

    def close_comparison(self) -> None:
        self.comparison = None
        if not self._busy():
            self.running = True

    def request_quit(self) -> None:
        if self._busy():
            self._set_status("Finalizando a operação antes de sair...")
            if self.job is not None and self.job.running:
                self.job.stop()
            self._quit_after_job = True
            return
        self._closing = True

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def _poll_jobs(self) -> None:
        if self.job is not None and not self._job_finalized:
            if not self.job.running:
                self._finalize_training(self.job)
        if self.comparison_job is not None and not self._comparison_finalized:
            if not self.comparison_job.running:
                self._finalize_comparison(self.comparison_job)
        if self._quit_after_job and not self._busy():
            self._closing = True

    def _finalize_training(self, job: TrainingJob) -> None:
        self._job_finalized = True
        if job.error is not None:
            self._set_status(f"Erro no treinamento: {job.error}")
            print("Erro no treinamento:", repr(job.error))
            traceback.print_exception(type(job.error), job.error, job.error.__traceback__)
            self.job = None
            return
        result = job.result or {}
        try:
            self.agent = QLearningAgent.load(self.config.model_path, self.config.agent)
        except (OSError, ValueError) as exc:
            self._set_status(f"Treino terminou, mas falhou ao carregar o agente: {exc}")
            self.job = None
            return
        self.env.agent = self.agent
        self.mode = "ai"
        self.seed = random.randint(1, 999_999)
        self.env.reset(
            seed=self.seed,
            mode=self.mode,
            training=False,
            episode_seconds=float("inf"),
        )
        self.accumulator = 0.0
        self.running = True
        completed = int(result.get("episodes_completed", 0))
        requested = int(result.get("episodes_requested", 0))
        epsilon = float(result.get("epsilon", self.agent.epsilon))
        suffix = " (interrompido, Q-table salva)" if result.get("stopped") else ""
        self._set_status(
            f"Treino concluído: {completed}/{requested} episódios{suffix}. "
            f"Agente carregado no modo IA · ε final {epsilon:.3f}."
        )
        self.job = None

    def _finalize_comparison(self, job: ComparisonJob) -> None:
        self._comparison_finalized = True
        if job.error is not None:
            self._set_status(f"Erro na avaliação: {job.error}")
            print("Erro na avaliação:", repr(job.error))
            traceback.print_exception(type(job.error), job.error, job.error.__traceback__)
        else:
            self.comparison = job.result
            if self.comparison:
                traditional = self.comparison["traditional"]
                ai = self.comparison["ai"]
                self._set_status(
                    "Avaliação concluída. Espera média: "
                    f"tradicional {traditional['avg_wait']:.2f}s · IA {ai['avg_wait']:.2f}s. "
                    "Pressione C para fechar."
                )
            else:
                self._set_status("Avaliação concluída sem resultados.")
        self.comparison_job = None
        self._comparison_pending = False

    def _update(self, dt_real: float) -> None:
        if self._closing:
            return
        if self.running and not self._busy():
            self.accumulator += min(dt_real, 0.10) * self.speed
            step = self.config.simulation.sim_dt
            steps = 0
            max_steps = 80
            while self.accumulator >= step and steps < max_steps:
                self.env.update(step)
                self.accumulator -= step
                steps += 1
            if steps >= max_steps:
                self.accumulator = 0.0

    def _ui(self) -> dict[str, Any]:
        training_info: dict[str, Any] | None = None
        if self.job is not None and self.job.running:
            training_info = dict(self.job.progress)
            if not training_info:
                training_info = {"episode": 0, "episodes": self.config.training_episodes}
        comparison_info = None
        if self.comparison_job is not None and self.comparison_job.running:
            comparison_info = self.comparison_job.progress
        return {
            "running": self.running,
            "speed": self.speed,
            "status": self.status,
            "training": training_info,
            "comparison_running": comparison_info,
            "comparison": self.comparison,
            "fps": self.fps,
        }

    def run(self) -> int:
        pygame = self.pygame
        while not self._closing:
            dt_real = self.clock.tick(self.rc.fps) / 1000.0
            if dt_real > 0:
                self.fps = self.fps * 0.9 + (1.0 / dt_real) * 0.1

            for event in pygame.event.get():
                self._handle_event(event)
                if self._closing:
                    break
            if self._closing:
                break

            self._poll_jobs()
            self._update(dt_real)

            alpha = 1.0
            if self.running and not self._busy():
                alpha = min(1.0, max(0.0, self.accumulator / self.config.simulation.sim_dt))
            self.renderer.draw(self.env, self.agent, self._ui(), alpha)
            pygame.display.flip()

        pygame.quit()
        return 0

    def _handle_event(self, event: Any) -> None:
        pygame = self.pygame
        if event.type == pygame.QUIT:
            self.request_quit()
            return
        if event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                if self.comparison is not None:
                    self.close_comparison()
                else:
                    self.request_quit()
                return
            if key == pygame.K_SPACE:
                self.toggle_pause()
            elif key == pygame.K_r:
                self.restart()
            elif key == pygame.K_m:
                self.toggle_mode()
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.change_speed(-1)
            elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.change_speed(1)
            elif key == pygame.K_t:
                self.toggle_training()
            elif key == pygame.K_s:
                self.save_agent()
            elif key == pygame.K_l:
                self.load_agent()
            elif key == pygame.K_c:
                if self.comparison is not None:
                    self.close_comparison()
                elif not (self.comparison_job and self.comparison_job.running):
                    self.start_comparison()
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            action = self.renderer.button_at(event.pos)
            if action:
                self._run_button(action)

    def _run_button(self, action: str) -> None:
        if action == "start":
            self.toggle_pause()
        elif action == "restart":
            self.restart()
        elif action == "mode":
            self.toggle_mode()
        elif action == "speed_down":
            self.change_speed(-1)
        elif action == "speed_up":
            self.change_speed(1)
        elif action == "train":
            self.toggle_training()
        elif action == "save":
            self.save_agent()
        elif action == "load":
            self.load_agent()
        elif action == "compare":
            self.start_comparison()
        elif action == "quit":
            self.request_quit()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autopilot",
        description="AutoPilot — simulador de trânsito com semáforos Q-Learning.",
    )
    parser.add_argument("--train", action="store_true", help="treinamento acelerado sem renderização")
    parser.add_argument("--evaluate", action="store_true", help="compara tradicional x IA")
    parser.add_argument("--episodes", type=int, default=None, help="episódios de treino")
    parser.add_argument("--eval-episodes", type=int, default=None, help="episódios por controlador na avaliação")
    parser.add_argument("--seconds", type=float, default=None, help="duração de cada episódio (s simulados)")
    parser.add_argument("--seed", type=int, default=None, help="seed base do treino")
    parser.add_argument("--eval-seed", type=int, default=None, help="seed base da avaliação")
    parser.add_argument("--model", type=str, default=None, help="caminho da Q-table (.npz)")
    parser.add_argument("--chart", type=str, default=None, help="caminho do gráfico comparativo (.png)")
    parser.add_argument("--report", type=str, default=None, help="caminho do relatório JSON")
    parser.add_argument("--resume", action="store_true", help="continua o treino a partir da Q-table informada")
    parser.add_argument("--no-chart", action="store_true", help="não gera o gráfico da avaliação")
    parser.add_argument("--gui-seed", type=int, default=None, help="seed inicial da interface gráfica")
    return parser


def _apply_paths(config: Config, args: argparse.Namespace) -> None:
    if args.model:
        config.model_path = Path(args.model).expanduser().resolve()
    if args.chart:
        config.chart_path = Path(args.chart).expanduser().resolve()
    if args.report:
        config.report_path = Path(args.report).expanduser().resolve()
    config.ensure_dirs()


def _run_training(config: Config, args: argparse.Namespace) -> QLearningAgent:
    agent: QLearningAgent | None = None
    fresh = True
    if args.resume:
        if config.model_path.exists():
            agent = QLearningAgent.load(config.model_path, config.agent)
            fresh = False
            print(f"Retomando treinamento de {config.model_path}...")
        else:
            print(f"Nenhum agente em {config.model_path}; iniciando do zero.")
    result = train(
        config,
        episodes=args.episodes,
        seconds=args.seconds,
        seed=args.seed,
        agent=agent,
        fresh=fresh,
        verbose=True,
    )
    return result["agent"]


def _run_evaluation(config: Config, args: argparse.Namespace, agent: QLearningAgent | None = None) -> int:
    if agent is None:
        if not config.model_path.exists():
            print(
                f"ERRO: Q-table não encontrada em {config.model_path}.\n"
                "Treine primeiro: python -m autopilot.main --train"
            )
            return 2
        try:
            agent = QLearningAgent.load(config.model_path, config.agent)
        except (OSError, ValueError) as exc:
            print(f"ERRO ao carregar o agente: {exc}")
            return 2
    if args.eval_seed is not None:
        config.evaluation_seed = args.eval_seed
    episodes = args.eval_episodes if args.eval_episodes is not None else None
    compare_controllers(
        config,
        agent=agent,
        episodes=episodes,
        seconds=args.seconds,
        verbose=True,
        save_chart=not args.no_chart,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config()
    _apply_paths(config, args)
    if args.seed is not None:
        config.training_seed = args.seed
    if args.eval_seed is not None:
        config.evaluation_seed = args.eval_seed

    try:
        if args.train:
            agent = _run_training(config, args)
            if args.evaluate:
                return _run_evaluation(config, args, agent)
            return 0
        if args.evaluate:
            return _run_evaluation(config, args)
        return App(config, seed=args.gui_seed).run()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        return 130
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependência"
        print(
            f"ERRO: módulo '{missing}' não encontrado. "
            "Instale as dependências com: python -m pip install -r autopilot/requirements.txt"
        )
        return 2
    except Exception as exc:
        print(f"ERRO inesperado: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
