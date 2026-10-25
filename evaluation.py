"""Avaliação comparativa entre o controle tradicional e a IA.

Regras de justiça:
* mesmas sementes;
* mesmos padrões de chegada (a simulação consome RNG apenas nas chegadas);
* mesma duração;
* o agente é avaliado de forma gananciosa, sem explorar e sem atualizar a
  Q-table.

O relatório e o gráfico são gerados apenas com os valores medidos.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .agent import QLearningAgent
from .config import Config
from .environment import TrafficEnvironment

ProgressCallback = Callable[[dict[str, Any]], None]

METRIC_LABELS = {
    "avg_wait": "Tempo médio de espera",
    "served": "Veículos atendidos",
    "avg_queue": "Fila média",
    "max_wait": "Maior espera registrada",
}


def run_episode(
    config: Config,
    *,
    seed: int,
    mode: str,
    agent: QLearningAgent | None = None,
    seconds: float | None = None,
) -> dict[str, Any]:
    """Executa um episódio fechado e devolve apenas métricas medidas."""
    env = TrafficEnvironment(config, agent=agent if mode == "ai" else None)
    env.reset(
        seed=seed,
        mode=mode,
        training=False,
        episode_seconds=config.training_seconds if seconds is None else seconds,
    )
    env.run()
    metrics = env.final_metrics()
    metrics["epsilon_used"] = 0.0 if agent is None else 0.0
    return metrics


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("nenhum episódio avaliado")
    return {
        "avg_wait": statistics.fmean(row["avg_wait"] for row in rows),
        "served_mean": statistics.fmean(row["served"] for row in rows),
        "served_total": float(sum(row["served"] for row in rows)),
        "avg_queue": statistics.fmean(row["avg_queue"] for row in rows),
        "max_wait": max(row["max_wait"] for row in rows),
        "arrived": float(sum(row["arrived"] for row in rows)),
        "reward_mean": statistics.fmean(row["episode_reward"] for row in rows),
        "switches_mean": statistics.fmean(row["switches"] for row in rows),
        "decisions_mean": statistics.fmean(row["decisions"] for row in rows),
    }


def compare_controllers(
    config: Config | None = None,
    *,
    agent: QLearningAgent | None = None,
    episodes: int | None = None,
    seeds: Iterable[int] | None = None,
    seconds: float | None = None,
    progress: ProgressCallback | None = None,
    verbose: bool = True,
    save_chart: bool = True,
) -> dict[str, Any]:
    """Roda a comparação tradicional x IA e salva relatório/gráfico."""
    cfg = config or Config()
    cfg.ensure_dirs()
    episodes = int(cfg.evaluation_episodes if episodes is None else episodes)
    seconds = float(cfg.training_seconds if seconds is None else seconds)
    if episodes <= 0:
        raise ValueError("episodes deve ser positivo")
    if seconds <= 0:
        raise ValueError("seconds deve ser positivo")
    if agent is None:
        raise ValueError(
            "a avaliação da IA exige um agente. Treine ou carregue uma Q-table."
        )
    if seeds is None:
        seed_list = [cfg.evaluation_seed + i for i in range(episodes)]
    else:
        seed_list = [int(seed) for seed in seeds]
        if not seed_list:
            raise ValueError("a lista de seeds não pode ser vazia")
        episodes = len(seed_list)

    # Auditoria: a avaliação não pode alterar a política nem a Q-table.
    table_before = agent.q_table.copy()
    updates_before = agent.total_updates
    decisions_before = agent.total_decisions

    traditional_rows: list[dict[str, Any]] = []
    ai_rows: list[dict[str, Any]] = []
    started = time.monotonic()

    if verbose:
        print("=" * 78)
        print("AutoPilot — avaliação comparativa")
        print(
            f"{episodes} episódios | {seconds:.0f}s cada | "
            f"seed inicial {seed_list[0]} | agente: {agent.summary()}"
        )
        print("A IA é avaliada sem exploração e sem atualizar a Q-table.")
        print("=" * 78, flush=True)

    for index, seed in enumerate(seed_list):
        traditional = run_episode(cfg, seed=seed, mode="traditional", seconds=seconds)
        ai = run_episode(cfg, seed=seed, mode="ai", agent=agent, seconds=seconds)
        traditional_rows.append(traditional)
        ai_rows.append(ai)

        if progress is not None:
            progress(
                {
                    "episode": index + 1,
                    "episodes": episodes,
                    "seed": seed,
                    "traditional": traditional,
                    "ai": ai,
                    "elapsed": time.monotonic() - started,
                }
            )
        if verbose:
            print(
                f"ep {index + 1:3d}/{episodes} seed={seed} | "
                f"trad: espera={traditional['avg_wait']:5.1f}s "
                f"atend={traditional['served']:3d} "
                f"fila={traditional['avg_queue']:4.1f} | "
                f"IA: espera={ai['avg_wait']:5.1f}s "
                f"atend={ai['served']:3d} fila={ai['avg_queue']:4.1f}",
                flush=True,
            )

    traditional_stats = _aggregate(traditional_rows)
    ai_stats = _aggregate(ai_rows)
    elapsed = time.monotonic() - started

    q_table_updates = agent.total_updates - updates_before
    decisions_during_eval = agent.total_decisions - decisions_before
    q_table_unchanged = bool(np.array_equal(agent.q_table, table_before))
    if q_table_updates != 0 or decisions_during_eval != 0 or not q_table_unchanged:
        raise RuntimeError(
            "a avaliação alterou o agente: "
            f"updates={q_table_updates}, decisões={decisions_during_eval}, "
            f"Q-table intacta={q_table_unchanged}"
        )

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes": episodes,
        "seconds_per_episode": seconds,
        "seeds": seed_list,
        "agent": {
            "episodes_trained": agent.episodes_trained,
            "epsilon_saved": agent.epsilon,
            "epsilon_used": 0.0,
            "exploration": False,
            "q_table_updates_during_eval": q_table_updates,
            "decisions_during_eval": decisions_during_eval,
            "q_table_unchanged": q_table_unchanged,
        },
        "traditional": traditional_stats,
        "ai": ai_stats,
        "episodes_detail": {
            "traditional": traditional_rows,
            "ai": ai_rows,
        },
        "elapsed_seconds": elapsed,
        "chart_path": str(cfg.chart_path) if save_chart else None,
        "report_path": str(cfg.report_path),
    }

    report_path = cfg.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove detalhes muito grandes? Não: o relatório é a fonte auditável.
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    if save_chart:
        try:
            _save_chart(cfg, result)
        except ImportError as exc:
            result["chart_error"] = str(exc)
            if verbose:
                print(f"AVISO: gráfico não gerado ({exc}).", flush=True)

    if verbose:
        _print_summary(result)
        print(f"Relatório: {report_path}")
        if save_chart:
            print(f"Gráfico:   {cfg.chart_path}")
        print("=" * 78, flush=True)

    return result


def _print_summary(result: dict[str, Any]) -> None:
    traditional = result["traditional"]
    ai = result["ai"]
    print("-" * 78)
    print(f"{'Métrica':<30}{'Tradicional':>16}{'IA':>16}{'Diferença':>14}")
    print("-" * 78)

    def line(label: str, key: str, unit: str = "", lower_is_better: bool = True) -> None:
        trad = float(traditional[key])
        ia = float(ai[key])
        diff = ia - trad
        if lower_is_better:
            marker = "melhor" if diff < -1e-9 else ("pior" if diff > 1e-9 else "igual")
        else:
            marker = "melhor" if diff > 1e-9 else ("pior" if diff < -1e-9 else "igual")
        print(
            f"{label:<30}{trad:>13.2f}{unit:<3}{ia:>13.2f}{unit:<3}"
            f"{diff:>+9.2f}{' ' + marker:>10}"
        )

    line("Tempo médio de espera", "avg_wait", "s", True)
    line("Veículos atendidos/ep.", "served_mean", "", False)
    line("Fila média (veículos)", "avg_queue", "", True)
    line("Maior espera registrada", "max_wait", "s", True)
    print("-" * 78)
    print(
        f"Atendidos no total: tradicional={traditional['served_total']:.0f} | "
        f"IA={ai['served_total']:.0f}"
    )
    print(
        f"Recompensa média IA por episódio: {ai['reward_mean']:.2f} | "
        f"trocas de fase/ep.: {ai['switches_mean']:.1f}"
    )


def _save_chart(config: Config, result: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib não está instalado") from exc

    traditional = result["traditional"]
    ai = result["ai"]
    metrics = [
        ("avg_wait", "Tempo médio de espera (s)", True),
        ("served_mean", "Veículos atendidos por episódio", False),
        ("avg_queue", "Fila média (veículos)", True),
        ("max_wait", "Maior espera registrada (s)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.suptitle(
        "AutoPilot — Controle Tradicional vs. IA (Q-Learning)\n"
        f"{result['episodes']} episódios · mesmas sementes · "
        f"{result['seconds_per_episode']:.0f}s por episódio",
        fontsize=13,
        fontweight="bold",
    )
    colors = ["#3b82f6", "#10b981"]
    for axis, (key, title, lower_is_better) in zip(axes.flat, metrics):
        values = [float(traditional[key]), float(ai[key])]
        bars = axis.bar(["Tradicional", "IA"], values, color=colors, width=0.55)
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=0.25, linestyle="--")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        upper = max(values) if max(values) > 0 else 1.0
        axis.set_ylim(0, upper * 1.18)
        subtitle = "menor é melhor" if lower_is_better else "maior é melhor"
        axis.text(
            0.99,
            0.94,
            subtitle,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#64748b",
        )

    fig.text(
        0.5,
        0.01,
        "Avaliação gananciosa (ε=0), sem atualização da Q-table. "
        "Métricas medidas na simulação.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    config.chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_signed(value: float, unit: str = "") -> str:
    suffix = unit
    return f"{value:+.2f}{suffix}"
