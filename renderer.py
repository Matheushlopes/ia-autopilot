"""Renderização Pygame do AutoPilot.

Toda a parte visual fica neste módulo. Ele lê o estado da simulação e do
agente, mas nunca altera física, recompensas ou Q-table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pygame

from .agent import KEEP, QLearningAgent
from .config import (
    AXIS_OF,
    Config,
    DIRECTION_LABELS,
    RenderConfig,
)
from .environment import TrafficEnvironment
from .vehicle import DIRECTION_VECTORS, Vehicle


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def _mix(color_a: tuple[int, int, int], color_b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def _format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _format_reward(value: float) -> str:
    return f"{value:+.2f}"


@dataclass
class Button:
    action: str
    label: str
    rect: pygame.Rect
    primary: bool = False
    danger: bool = False
    enabled: bool = True


class Renderer:
    """Desenha o cruzamento, o HUD e o painel lateral."""

    def __init__(self, config: Config, render: RenderConfig | None = None) -> None:
        self.config = config
        self.rc = render or RenderConfig()
        pygame.font.init()
        self.fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self.buttons: list[Button] = []
        self.clock_font = self._font(11, True)
        self._static = self._build_static()
        self._hud_bar = self._build_hud_bar()

    # ------------------------------------------------------------------
    # Fontes e desenho básico
    # ------------------------------------------------------------------
    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self.fonts:
            path = pygame.font.match_font("dejavusans,liberationsans,arial")
            if path:
                self.fonts[key] = pygame.font.Font(path, size)
                if bold:
                    self.fonts[key].set_bold(True)
            else:
                self.fonts[key] = pygame.font.SysFont(None, size, bold=bold)
        return self.fonts[key]

    def _text(
        self,
        surface: pygame.Surface,
        text: str,
        size: int,
        color: tuple[int, int, int],
        pos: tuple[float, float],
        *,
        anchor: str = "topleft",
        bold: bool = False,
    ) -> pygame.Rect:
        rendered = self._font(size, bold).render(str(text), True, color)
        rect = rendered.get_rect(**{anchor: pos})
        surface.blit(rendered, rect)
        return rect

    def _round_rect(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        radius: int = 10,
        *,
        width: int = 0,
        border: tuple[int, int, int] | None = None,
    ) -> None:
        pygame.draw.rect(surface, color, rect, border_radius=radius, width=width)
        if border is not None:
            pygame.draw.rect(surface, border, rect, width=1, border_radius=radius)

    def _alpha_rect(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        alpha: int,
        radius: int = 0,
    ) -> None:
        layer = pygame.Surface((max(1, rect.width), max(1, rect.height)), pygame.SRCALPHA)
        pygame.draw.rect(
            layer,
            (*color, max(0, min(255, alpha))),
            layer.get_rect(),
            border_radius=radius,
        )
        surface.blit(layer, rect.topleft)

    def _dashed_line(
        self,
        surface: pygame.Surface,
        color: tuple[int, int, int],
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        dash: float = 10,
        gap: float = 12,
        width: int = 2,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        position = 0.0
        while position < length:
            segment_end = min(length, position + dash)
            pygame.draw.line(
                surface,
                color,
                (x1 + ux * position, y1 + uy * position),
                (x1 + ux * segment_end, y1 + uy * segment_end),
                width,
            )
            position += dash + gap

    def _wrap(self, text: str, size: int, max_width: int, max_lines: int = 2) -> list[str]:
        font = self._font(size)
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
                if len(lines) >= max_lines - 1:
                    break
        if current and len(lines) < max_lines:
            lines.append(current)
        if not lines:
            lines = [""]
        if len(lines) == max_lines and font.size(" ".join(words))[0] > max_width:
            last = lines[-1]
            while last and font.size(last + "…")[0] > max_width:
                last = last[:-1]
            lines[-1] = last + "…"
        return lines

    # ------------------------------------------------------------------
    # Camada estática
    # ------------------------------------------------------------------
    def _build_static(self) -> pygame.Surface:
        rc = self.rc
        surface = pygame.Surface((rc.main_width, rc.height)).convert()
        # Gradiente vertical suave.
        top = (16, 22, 33)
        bottom = (9, 13, 20)
        for y in range(rc.height):
            t = y / max(1, rc.height - 1)
            pygame.draw.line(surface, _mix(top, bottom, t), (0, y), (rc.main_width, y))

        # Grade sutil.
        grid = (23, 31, 44)
        for x in range(0, rc.main_width, 44):
            pygame.draw.line(surface, grid, (x, 0), (x, rc.height), 1)
        for y in range(0, rc.height, 44):
            pygame.draw.line(surface, grid, (0, y), (rc.main_width, y), 1)

        cx, cy = rc.center
        scale = rc.scale
        half = self.config.simulation.road_half_width * scale
        lane = self.config.simulation.lane_offset * scale

        # Calçadas e asfalto.
        sidewalk_extra = 12
        pygame.draw.rect(
            surface,
            rc.sidewalk,
            pygame.Rect(cx - half - sidewalk_extra, 0, 2 * (half + sidewalk_extra), rc.height),
        )
        pygame.draw.rect(
            surface,
            rc.sidewalk,
            pygame.Rect(0, cy - half - sidewalk_extra, rc.main_width, 2 * (half + sidewalk_extra)),
        )
        pygame.draw.rect(
            surface,
            rc.road_edge,
            pygame.Rect(cx - half, 0, 2 * half, rc.height),
        )
        pygame.draw.rect(
            surface,
            rc.road_edge,
            pygame.Rect(0, cy - half, rc.main_width, 2 * half),
        )
        pygame.draw.rect(
            surface,
            rc.road,
            pygame.Rect(cx - half + 3, 0, 2 * half - 6, rc.height),
        )
        pygame.draw.rect(
            surface,
            rc.road,
            pygame.Rect(0, cy - half + 3, rc.main_width, 2 * half - 6),
        )

        # Textura de asfalto (determinística).
        import random as _random

        rng = _random.Random(7)
        for _ in range(420):
            x = rng.randint(0, rc.main_width - 1)
            y = rng.randint(0, rc.height - 1)
            in_vertical = abs(x - cx) < half
            in_horizontal = abs(y - cy) < half
            if in_vertical or in_horizontal:
                tone = rng.choice((-5, 5, 7, 9))
                color = tuple(max(0, min(255, c + tone)) for c in rc.road)
                surface.set_at((x, y), color)

        # Linhas centrais tracejadas (amarelo suave), fora do cruzamento.
        gap_intersection = half + 6
        mark_color = (187, 173, 118)
        self._dashed_line(
            surface, mark_color, (cx, 0), (cx, cy - gap_intersection), dash=14, gap=14, width=2
        )
        self._dashed_line(
            surface,
            mark_color,
            (cx, cy + gap_intersection),
            (cx, rc.height),
            dash=14,
            gap=14,
            width=2,
        )
        self._dashed_line(
            surface, mark_color, (0, cy), (cx - gap_intersection, cy), dash=14, gap=14, width=2
        )
        self._dashed_line(
            surface,
            mark_color,
            (cx + gap_intersection, cy),
            (rc.main_width, cy),
            dash=14,
            gap=14,
            width=2,
        )

        # Bordas de faixa.
        edge = (92, 105, 122)
        for x in (cx - half + 4, cx + half - 4):
            pygame.draw.line(surface, edge, (x, 0), (x, cy - half), 1)
            pygame.draw.line(surface, edge, (x, cy + half), (x, rc.height), 1)
        for y in (cy - half + 4, cy + half - 4):
            pygame.draw.line(surface, edge, (0, y), (cx - half, y), 1)
            pygame.draw.line(surface, edge, (cx + half, y), (rc.main_width, y), 1)

        # Faixas de pedestres.
        self._crosswalk(surface, cx, cy, half, scale)

        # Linhas de retenção (apenas no sentido de chegada).
        white = (226, 232, 240)
        bar = 3
        pygame.draw.rect(surface, white, pygame.Rect(int(cx - half), int(cy - half), int(half - 2 * scale), bar))
        pygame.draw.rect(surface, white, pygame.Rect(int(cx + 2 * scale), int(cy + half - bar), int(half - 2 * scale), bar))
        pygame.draw.rect(surface, white, pygame.Rect(int(cx + half - bar), int(cy - half), bar, int(half - 2 * scale)))
        pygame.draw.rect(surface, white, pygame.Rect(int(cx - half), int(cy + 2 * scale), bar, int(half - 2 * scale)))

        # Setas indicando o sentido das aproximações.
        arrow = (168, 181, 198)
        self._arrow(surface, (cx - lane, cy - 92), (0, 1), arrow)
        self._arrow(surface, (cx + lane, cy + 92), (0, -1), arrow)
        self._arrow(surface, (cx + 92, cy - lane), (-1, 0), arrow)
        self._arrow(surface, (cx - 92, cy + lane), (1, 0), arrow)

        # Rótulos das vias.
        label_color = (112, 128, 150)
        self._text(surface, "NORTE", 9, label_color, (cx, 64), anchor="midtop", bold=True)
        self._text(surface, "SUL", 9, label_color, (cx, rc.height - 16), anchor="midbottom", bold=True)
        west = self._font(9, True).render("OESTE", True, label_color)
        west = pygame.transform.rotate(west, 90)
        surface.blit(west, west.get_rect(center=(15, cy)))
        east = self._font(9, True).render("LESTE", True, label_color)
        east = pygame.transform.rotate(east, -90)
        surface.blit(east, east.get_rect(center=(rc.main_width - 15, cy)))
        return surface

    def _crosswalk(self, surface: pygame.Surface, cx: float, cy: float, half: float, scale: float) -> None:
        color = (190, 199, 210)
        stripe = 6
        gap = 6
        # Norte e Sul.
        x_start = cx - half + 4
        x_end = cx + half - 4
        for y in (cy - half - 13, cy + half + 2):
            x = x_start
            while x + stripe <= x_end:
                pygame.draw.rect(surface, color, pygame.Rect(int(x), int(y), stripe, 11))
                x += stripe + gap
        # Leste e Oeste.
        y_start = cy - half + 4
        y_end = cy + half - 4
        for x in (cx - half - 13, cx + half + 2):
            y = y_start
            while y + stripe <= y_end:
                pygame.draw.rect(surface, color, pygame.Rect(int(x), int(y), 11, stripe))
                y += stripe + gap

    def _arrow(
        self,
        surface: pygame.Surface,
        center: tuple[float, float],
        direction: tuple[float, float],
        color: tuple[int, int, int],
    ) -> None:
        x, y = center
        ux, uy = direction
        px, py = -uy, ux
        length = 12
        head = 7
        points = [
            (x + ux * length, y + uy * length),
            (x + px * head, y + py * head),
            (x - px * head, y - py * head),
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.line(
            surface,
            color,
            (x - ux * length, y - uy * length),
            (x + ux * (length - 3), y + uy * (length - 3)),
            2,
        )

    def _build_hud_bar(self) -> pygame.Surface:
        bar = pygame.Surface((self.rc.main_width, 58), pygame.SRCALPHA)
        pygame.draw.rect(bar, (8, 12, 19, 214), bar.get_rect())
        pygame.draw.line(bar, (33, 45, 61, 255), (0, 57), (self.rc.main_width, 57), 1)
        return bar

    # ------------------------------------------------------------------
    # API de desenho
    # ------------------------------------------------------------------
    def button_at(self, pos: tuple[int, int]) -> str | None:
        for button in reversed(self.buttons):
            if button.enabled and button.rect.collidepoint(pos):
                return button.action
        return None

    def draw(
        self,
        env: TrafficEnvironment,
        agent: QLearningAgent | None,
        ui: dict[str, Any],
        alpha: float = 1.0,
    ) -> None:
        screen = pygame.display.get_surface()
        if screen is None:
            raise RuntimeError("display Pygame não inicializado")
        rc = self.rc
        screen.fill(rc.background)
        screen.blit(self._static, (0, 0))

        self._draw_traffic_lights(screen, env)
        self._draw_vehicles(screen, env, alpha)
        screen.blit(self._hud_bar, (0, 0))
        self._draw_hud(screen, env, agent, ui)
        self._draw_info_card(screen, env, ui)
        self._draw_panel(screen, env, agent, ui)

        if ui.get("training"):
            self._draw_training_overlay(screen, ui["training"])
        if ui.get("comparison_running") is not None:
            self._draw_evaluation_overlay(screen, ui.get("comparison_running") or {})
        if ui.get("comparison"):
            self._draw_comparison_overlay(screen, ui["comparison"])
        if (
            not ui.get("running", True)
            and not ui.get("training")
            and not ui.get("comparison")
            and ui.get("comparison_running") is None
        ):
            self._draw_pause_overlay(screen)

    # ------------------------------------------------------------------
    # Cruzamento
    # ------------------------------------------------------------------
    def _world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.rc.center
        return (cx + x * self.rc.scale, cy + y * self.rc.scale)

    def _draw_traffic_lights(self, screen: pygame.Surface, env: TrafficEnvironment) -> None:
        rc = self.rc
        cx, cy = rc.center
        half = self.config.simulation.road_half_width * rc.scale
        light = env.sim.light
        positions = [
            ("north", (cx - half - 36, cy - half - 16), (cx - half - 6, cy - half - 4), "N"),
            ("south", (cx + half + 18, cy + half - 24), (cx + half + 6, cy + half + 4), "S"),
            ("east", (cx + half + 18, cy - half - 16), (cx + half + 6, cy - half - 4), "L"),
            ("west", (cx - half - 36, cy + half - 24), (cx - half - 6, cy + half + 4), "O"),
        ]
        for direction, box_xy, pole_start, label in positions:
            axis = AXIS_OF[direction]
            signal = light.signal(axis)
            color = {
                "green": rc.green,
                "yellow": rc.yellow,
                "red": rc.red,
            }[signal]
            box = pygame.Rect(int(box_xy[0]), int(box_xy[1]), 18, 42)
            pygame.draw.line(screen, (63, 74, 90), pole_start, (box.centerx, box.bottom), 2)
            self._alpha_rect(screen, box.inflate(6, 6), (0, 0, 0), 90, radius=7)
            self._round_rect(screen, box, (10, 15, 23), radius=5, border=(58, 72, 90))

            lights = [
                ("red", rc.red, box.y + 9),
                ("yellow", rc.yellow, box.y + 21),
                ("green", rc.green, box.y + 33),
            ]
            for name, lcolor, y in lights:
                active = name == signal
                if active:
                    self._glow(screen, (box.centerx, y), 10, lcolor, alpha=95)
                    pygame.draw.circle(screen, lcolor, (box.centerx, y), 5)
                    pygame.draw.circle(screen, (255, 255, 255), (box.centerx - 1, y - 1), 1)
                else:
                    dim = _mix((18, 24, 34), lcolor, 0.16)
                    pygame.draw.circle(screen, dim, (box.centerx, y), 4)
            self._text(
                screen,
                label,
                9,
                _mix(rc.text_muted, color, 0.55),
                (box.centerx, box.bottom + 4),
                anchor="midtop",
                bold=True,
            )

        # Pequena indicação de eixos no asfalto, perto do centro.
        self._text(
            screen,
            "N/S",
            8,
            (120, 133, 152),
            (cx - 20, cy + 5),
            anchor="topright",
            bold=True,
        )
        self._text(
            screen,
            "L/O",
            8,
            (120, 133, 152),
            (cx + 20, cy + 5),
            anchor="topleft",
            bold=True,
        )

    def _glow(
        self,
        screen: pygame.Surface,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        *,
        alpha: int = 90,
    ) -> None:
        size = radius * 4
        layer = pygame.Surface((size, size), pygame.SRCALPHA)
        for index in range(4, 0, -1):
            layer_alpha = int(alpha * (index / 4) * 0.35)
            pygame.draw.circle(
                layer,
                (*color, layer_alpha),
                (size // 2, size // 2),
                radius + index * 2,
            )
        screen.blit(layer, (center[0] - size // 2, center[1] - size // 2))

    def _draw_vehicles(self, screen: pygame.Surface, env: TrafficEnvironment, alpha: float) -> None:
        for vehicle in env.sim.vehicles:
            self._draw_vehicle(screen, vehicle, alpha)

    def _draw_vehicle(self, screen: pygame.Surface, vehicle: Vehicle, alpha: float) -> None:
        rc = self.rc
        x_world, y_world = vehicle.interpolated_position(self.config.simulation, alpha)
        sx, sy = self._world_to_screen(x_world, y_world)
        ux, uy = vehicle.heading()
        px, py = -uy, ux
        half_l = vehicle.length * rc.scale / 2.0
        half_w = vehicle.width * rc.scale / 2.0

        def corners(center: tuple[float, float], length: float, width: float) -> list[tuple[float, float]]:
            cx, cy = center
            return [
                (cx + ux * length - px * width, cy + uy * length - py * width),
                (cx + ux * length + px * width, cy + uy * length + py * width),
                (cx - ux * length + px * width, cy - uy * length + py * width),
                (cx - ux * length - px * width, cy - uy * length - py * width),
            ]

        # Sombra.
        pygame.draw.polygon(
            screen,
            (10, 14, 21),
            corners((sx + 2, sy + 3), half_l, half_w),
        )
        body = corners((sx, sy), half_l, half_w)
        pygame.draw.polygon(screen, vehicle.color, body)
        pygame.draw.polygon(screen, _shade(vehicle.color, 0.62), body, 1)

        # Teto e para-brisa.
        roof = corners((sx, sy), half_l * 0.52, half_w * 0.72)
        pygame.draw.polygon(screen, _mix(vehicle.color, (235, 245, 255), 0.22), roof)
        windshield = corners(
            (sx + ux * half_l * 0.30, sy + uy * half_l * 0.30),
            half_l * 0.12,
            half_w * 0.62,
        )
        pygame.draw.polygon(screen, (178, 214, 238), windshield)

        # Luzes de freio quando o veículo está parado/desacelerando.
        if vehicle.speed < 1.8:
            rear_x = sx - ux * half_l
            rear_y = sy - uy * half_l
            for side in (-1, 1):
                lx = rear_x + px * half_w * 0.55 * side
                ly = rear_y + py * half_w * 0.55 * side
                pygame.draw.circle(screen, (255, 72, 72), (int(lx), int(ly)), 2)
        elif vehicle.speed > 2.5:
            front_x = sx + ux * half_l
            front_y = sy + uy * half_l
            for side in (-1, 1):
                lx = front_x + px * half_w * 0.55 * side
                ly = front_y + py * half_w * 0.55 * side
                pygame.draw.circle(screen, (255, 240, 190), (int(lx), int(ly)), 1)

    # ------------------------------------------------------------------
    # HUD superior
    # ------------------------------------------------------------------
    def _draw_hud(
        self,
        screen: pygame.Surface,
        env: TrafficEnvironment,
        agent: QLearningAgent | None,
        ui: dict[str, Any],
    ) -> None:
        rc = self.rc
        self._text(screen, "AUTOPILOT", 18, rc.text, (18, 12), bold=True)
        pygame.draw.circle(screen, rc.accent, (26, 44), 3)
        self._text(screen, "SIMULAÇÃO DE TRÁFEGO · Q-LEARNING", 8, rc.text_muted, (34, 39), bold=True)

        light = env.sim.light
        color = rc.green if light.is_green else (rc.yellow if light.is_yellow else rc.red)
        label = light.phase_label()
        if light.is_green:
            timer = f"{light.green_elapsed:04.1f}s"
        else:
            timer = f"{light.elapsed:04.1f}s"
        width = 278
        pill = pygame.Rect(self.rc.main_width // 2 - width // 2, 11, width, 36)
        self._round_rect(
            screen,
            pill,
            _mix((17, 24, 35), color, 0.13),
            radius=11,
            border=_mix((17, 24, 35), color, 0.55),
        )
        self._glow(screen, (pill.x + 18, pill.centery), 8, color, alpha=110)
        pygame.draw.circle(screen, color, (pill.x + 18, pill.centery), 4)
        self._text(screen, label, 11, color, (pill.x + 30, pill.centery - 7), bold=True)
        self._text(screen, timer, 11, rc.text_muted, (pill.right - 14, pill.centery - 7), anchor="topright")

        mode_ai = env.mode == "ai"
        mode_color = rc.green if mode_ai else rc.accent_blue
        mode_text = "IA · Q-LEARNING" if mode_ai else "TRADICIONAL"
        mode_width = 126 if mode_ai else 112
        mode_rect = pygame.Rect(rc.main_width - mode_width - 18, 14, mode_width, 28)
        self._round_rect(
            screen,
            mode_rect,
            _mix((17, 24, 35), mode_color, 0.14),
            radius=8,
            border=_mix((17, 24, 35), mode_color, 0.48),
        )
        self._text(
            screen,
            mode_text,
            9,
            mode_color,
            mode_rect.center,
            anchor="center",
            bold=True,
        )

    # ------------------------------------------------------------------
    # Cartão inferior de decisão
    # ------------------------------------------------------------------
    def _draw_info_card(
        self,
        screen: pygame.Surface,
        env: TrafficEnvironment,
        ui: dict[str, Any],
    ) -> None:
        rc = self.rc
        card = pygame.Rect(16, rc.height - 76, 334, 60)
        self._alpha_rect(screen, card, (8, 12, 19), 190, radius=12)
        pygame.draw.rect(screen, (36, 48, 65), card, 1, border_radius=12)
        if env.mode == "ai":
            action = "MANTER FASE" if env.last_action == KEEP else "TROCAR FASE"
            action_color = rc.accent_blue if env.last_action == KEEP else rc.yellow
            self._text(screen, "ÚLTIMA DECISÃO DA IA", 8, rc.text_muted, (30, card.y + 9), bold=True)
            self._text(screen, action, 13, action_color, (30, card.y + 21), bold=True)
            self._text(
                screen,
                f"recompensa {_format_reward(env.last_reward)}",
                9,
                rc.text_muted,
                (30, card.y + 41),
            )
            self._text(
                screen,
                f"decisões {env.decisions} · trocas {env.switches}",
                9,
                rc.text_muted,
                (card.right - 14, card.y + 41),
                anchor="topright",
            )
        else:
            self._text(screen, "CONTROLE TRADICIONAL", 8, rc.text_muted, (30, card.y + 9), bold=True)
            self._text(screen, "TEMPORIZAÇÃO FIXA", 13, rc.accent_blue, (30, card.y + 21), bold=True)
            self._text(
                screen,
                f"{self.config.lights.green_duration:g}s verde · "
                f"{self.config.lights.yellow_duration:g}s amarelo · "
                f"{self.config.lights.all_red_duration:g}s vermelho geral",
                9,
                rc.text_muted,
                (30, card.y + 41),
            )

    # ------------------------------------------------------------------
    # Painel lateral
    # ------------------------------------------------------------------
    def _draw_panel(
        self,
        screen: pygame.Surface,
        env: TrafficEnvironment,
        agent: QLearningAgent | None,
        ui: dict[str, Any],
    ) -> None:
        rc = self.rc
        stats = env.panel_stats()
        panel = pygame.Rect(rc.main_width, 0, rc.panel_width, rc.height)
        pygame.draw.rect(screen, rc.panel_bg, panel)
        pygame.draw.line(screen, (34, 47, 64), (panel.x, 0), (panel.x, panel.height), 1)

        pad = 24
        x = panel.x + pad
        content_width = rc.panel_width - 2 * pad

        # Cabeçalho.
        logo = pygame.Rect(x, 20, 40, 40)
        self._round_rect(screen, logo, (24, 35, 50), radius=11, border=(47, 63, 83))
        for index, color in enumerate(((248, 113, 113), (251, 191, 36), (52, 211, 153))):
            pygame.draw.circle(screen, _mix((24, 35, 50), color, 0.75), (logo.centerx, logo.y + 11 + index * 9), 3)
        self._text(screen, "AutoPilot", 24, rc.text, (x + 52, 18), bold=True)
        self._text(screen, "Semáforos com aprendizado por reforço", 9, rc.text_muted, (x + 53, 47))

        # Cartão de modo.
        mode_ai = env.mode == "ai"
        mode_color = rc.green if mode_ai else rc.accent_blue
        mode_card = pygame.Rect(x, 84, content_width, 58)
        self._round_rect(
            screen,
            mode_card,
            _mix(rc.panel_card, mode_color, 0.08),
            radius=12,
            border=_mix(rc.panel_card, mode_color, 0.38),
        )
        self._text(screen, "MODO DE CONTROLE", 8, rc.text_muted, (x + 14, mode_card.y + 9), bold=True)
        self._text(
            screen,
            "INTELIGÊNCIA ARTIFICIAL" if mode_ai else "TEMPORIZAÇÃO FIXA",
            14,
            mode_color,
            (x + 14, mode_card.y + 23),
            bold=True,
        )
        badge = pygame.Rect(mode_card.right - 54, mode_card.y + 18, 40, 22)
        self._round_rect(screen, badge, _mix(rc.panel_card, mode_color, 0.20), radius=6)
        self._text(screen, "IA" if mode_ai else "FIXO", 8, mode_color, badge.center, anchor="center", bold=True)

        # Linha do agente.
        training = ui.get("training")
        if training:
            trained_text = f"treinando · ep. {training.get('episode', 0)}/{training.get('episodes', '?')}"
            agent_color = rc.yellow
        elif agent is not None and agent.is_trained:
            trained_text = f"treinado · {agent.episodes_trained} ep."
            agent_color = rc.green
        else:
            trained_text = "NÃO TREINADO · pressione T"
            agent_color = rc.red
        self._text(screen, "AGENTE", 8, rc.text_muted, (x, 154), bold=True)
        self._text(screen, trained_text, 10, agent_color, (x + 58, 153), bold=True)

        # Telemetria.
        self._text(screen, "TELEMETRIA", 8, rc.text_muted, (x, 180), bold=True)
        reward = float(stats.get("episode_reward", 0.0))
        epsilon = 0.0
        if training:
            epsilon = float(training.get("epsilon", 0.0))
        elif agent is not None and env.mode == "ai":
            epsilon = 0.0
        episodes = agent.episodes_trained if agent is not None else 0
        if training:
            episodes = int(training.get("completed", training.get("episode", 0)))

        metrics = [
            ("TEMPO", _format_time(float(stats.get("time", 0.0)))),
            ("VELOCIDADE", f"{float(ui.get('speed', 1.0)):.1f}x"),
            ("VEÍCULOS", str(int(stats.get("active", 0)))),
            ("ATRAVESSARAM", str(int(stats.get("crossed", 0)))),
            ("ESPERA MÉDIA", f"{float(stats.get('avg_wait', 0.0)):.1f}s"),
            ("MAIOR ESPERA", f"{float(stats.get('max_wait', 0.0)):.1f}s"),
            ("FILA N/S", str(int(stats.get("queue_ns", 0)))),
            ("FILA L/O", str(int(stats.get("queue_ew", 0)))),
            ("RECOMPENSA", _format_reward(reward) if env.mode == "ai" else "—"),
            ("EPSILON", f"{epsilon:.3f}" if env.mode == "ai" else "—"),
            ("EP. TREINADOS", str(episodes)),
            ("DECISÕES", str(int(stats.get("decisions", 0))) if env.mode == "ai" else "—"),
        ]

        card_w = (content_width - 12) // 2
        card_h = 45
        gap = 6
        start_y = 198
        for index, (label, value) in enumerate(metrics):
            col = index % 2
            row = index // 2
            rect = pygame.Rect(x + col * (card_w + 12), start_y + row * (card_h + gap), card_w, card_h)
            highlight = label in ("RECOMPENSA", "EPSILON")
            self._round_rect(
                screen,
                rect,
                _mix(rc.panel_card, rc.accent, 0.035) if highlight else rc.panel_card,
                radius=9,
                border=(35, 48, 65),
            )
            self._text(screen, label, 7, rc.text_muted, (rect.x + 10, rect.y + 7), bold=True)
            value_color = rc.text
            if label == "RECOMPENSA" and env.mode == "ai":
                value_color = rc.green if reward >= 0 else rc.red
            if label == "EPSILON" and env.mode == "ai":
                value_color = rc.accent_blue
            self._text(screen, value, 14, value_color, (rect.x + 10, rect.y + 20), bold=True)

        # Controles.
        divider_y = start_y + 6 * (card_h + gap) + 2
        pygame.draw.line(screen, (34, 47, 64), (x, divider_y), (x + content_width, divider_y), 1)
        self._text(screen, "CONTROLES", 8, rc.text_muted, (x, divider_y + 10), bold=True)

        button_w = (content_width - 12) // 2
        button_h = 31
        button_gap = 6
        button_y = divider_y + 26
        running = bool(ui.get("running", True))
        training_active = training is not None
        definitions = [
            ("start", "Pausar" if running else "Iniciar", True, False),
            ("restart", "Reiniciar", False, False),
            ("mode", "Modo tradicional" if mode_ai else "Modo IA", False, False),
            ("speed_down", "Velocidade −", False, False),
            ("speed_up", "Velocidade +", False, False),
            ("train", "Parar treino" if training_active else "Treinar IA", False, training_active),
            ("save", "Salvar agente", False, False),
            ("load", "Carregar agente", False, False),
            ("compare", "Comparar", False, False),
            ("quit", "Sair", False, False),
        ]
        self.buttons = []
        mouse = pygame.mouse.get_pos()
        for index, (action, label, primary, danger) in enumerate(definitions):
            col = index % 2
            row = index // 2
            rect = pygame.Rect(
                x + col * (button_w + 12),
                button_y + row * (button_h + button_gap),
                button_w,
                button_h,
            )
            button = Button(action=action, label=label, rect=rect, primary=primary, danger=danger)
            self.buttons.append(button)
            self._draw_button(screen, button, mouse)

        # Rodapé / status.
        footer_y = button_y + 5 * (button_h + button_gap) + 5
        status = str(ui.get("status", ""))
        lines = self._wrap(status, 9, content_width, max_lines=2)
        for index, line in enumerate(lines):
            self._text(screen, line, 9, rc.text, (x, footer_y + index * 13))
        self._text(
            screen,
            "ESPAÇO pausar · M modo · T treinar · S/L agente",
            8,
            rc.text_muted,
            (x, rc.height - 28),
        )
        self._text(
            screen,
            f"passo fixo {self.config.simulation.sim_dt:.2f}s · seed {env.seed}",
            8,
            _mix(rc.text_muted, rc.panel_bg, 0.35),
            (x, rc.height - 15),
        )

    def _draw_button(self, screen: pygame.Surface, button: Button, mouse: tuple[int, int]) -> None:
        rc = self.rc
        hover = button.rect.collidepoint(mouse)
        if button.danger:
            base = (57, 27, 32)
            border = (120, 53, 62)
            text_color = rc.red
        elif button.primary:
            base = _mix(rc.panel_card, rc.accent_blue, 0.22)
            border = _mix(rc.panel_card, rc.accent_blue, 0.55)
            text_color = (219, 234, 254)
        else:
            base = (29, 40, 55) if hover else (24, 34, 47)
            border = (66, 85, 108) if hover else (42, 57, 75)
            text_color = rc.text if hover else (207, 218, 233)
        if hover:
            base = _shade(base, 1.12)
        self._round_rect(screen, button.rect, base, radius=8, border=border)
        self._text(
            screen,
            button.label,
            10,
            text_color,
            button.rect.center,
            anchor="center",
            bold=True,
        )

    # ------------------------------------------------------------------
    # Sobreposições
    # ------------------------------------------------------------------
    def _draw_pause_overlay(self, screen: pygame.Surface) -> None:
        rc = self.rc
        layer = pygame.Surface((rc.main_width, rc.height), pygame.SRCALPHA)
        layer.fill((5, 8, 13, 120))
        screen.blit(layer, (0, 0))
        card = pygame.Rect(rc.main_width // 2 - 150, rc.height // 2 - 58, 300, 116)
        self._alpha_rect(screen, card, (13, 19, 28), 232, radius=16)
        pygame.draw.rect(screen, (58, 75, 97), card, 1, border_radius=16)
        self._text(screen, "PAUSADO", 26, self.rc.text, (card.centerx, card.y + 26), anchor="midtop", bold=True)
        self._text(
            screen,
            "Espaço para continuar",
            10,
            self.rc.text_muted,
            (card.centerx, card.y + 70),
            anchor="midtop",
        )

    def _draw_training_overlay(self, screen: pygame.Surface, info: dict[str, Any]) -> None:
        rc = self.rc
        layer = pygame.Surface((rc.main_width, rc.height), pygame.SRCALPHA)
        layer.fill((5, 8, 13, 175))
        screen.blit(layer, (0, 0))
        card = pygame.Rect(rc.main_width // 2 - 290, rc.height // 2 - 132, 580, 264)
        self._alpha_rect(screen, card, (15, 22, 32), 240, radius=18)
        pygame.draw.rect(screen, (54, 72, 94), card, 1, border_radius=18)
        self._text(screen, "TREINANDO AGENTE", 20, rc.green, (card.centerx, card.y + 24), anchor="midtop", bold=True)
        self._text(
            screen,
            "simulação acelerada, sem renderização",
            9,
            rc.text_muted,
            (card.centerx, card.y + 52),
            anchor="midtop",
        )
        episode = int(info.get("episode", 0))
        episodes = int(info.get("episodes", 1)) or 1
        progress = min(1.0, episode / episodes)
        bar = pygame.Rect(card.x + 40, card.y + 86, card.width - 80, 12)
        self._round_rect(screen, bar, (28, 39, 53), radius=6)
        fill = pygame.Rect(bar.x, bar.y, max(0, int(bar.width * progress)), bar.height)
        if fill.width > 0:
            self._round_rect(screen, fill, rc.green, radius=6)

        self._text(
            screen,
            f"{episode} / {episodes} episódios",
            13,
            rc.text,
            (bar.x, bar.bottom + 12),
            bold=True,
        )
        self._text(
            screen,
            f"{progress * 100:5.1f}%",
            13,
            rc.green,
            (bar.right, bar.bottom + 12),
            anchor="topright",
            bold=True,
        )
        stats = [
            ("recompensa", f"{float(info.get('reward', 0.0)):+.2f}"),
            ("espera média", f"{float(info.get('avg_wait', 0.0)):.1f}s"),
            ("atendidos", str(int(info.get("served", 0)))),
            ("epsilon", f"{float(info.get('epsilon', 0.0)):.3f}"),
        ]
        for index, (label, value) in enumerate(stats):
            col = index % 2
            row = index // 2
            bx = card.x + 40 + col * (card.width - 80) // 2
            by = card.y + 142 + row * 34
            self._text(screen, label, 8, rc.text_muted, (bx, by), bold=True)
            self._text(screen, value, 12, rc.text, (bx + 90, by - 2), bold=True)
        self._text(
            screen,
            "Pressione T para interromper com segurança (a Q-table será salva)",
            9,
            rc.yellow,
            (card.centerx, card.bottom - 30),
            anchor="midtop",
        )
        self._text(
            screen,
            "O progresso também aparece no terminal",
            8,
            rc.text_muted,
            (card.centerx, card.bottom - 15),
            anchor="midtop",
        )

    def _draw_evaluation_overlay(self, screen: pygame.Surface, info: dict[str, Any]) -> None:
        rc = self.rc
        layer = pygame.Surface((rc.main_width, rc.height), pygame.SRCALPHA)
        layer.fill((5, 8, 13, 160))
        screen.blit(layer, (0, 0))
        card = pygame.Rect(rc.main_width // 2 - 220, rc.height // 2 - 78, 440, 156)
        self._alpha_rect(screen, card, (15, 22, 32), 240, radius=16)
        pygame.draw.rect(screen, (54, 72, 94), card, 1, border_radius=16)
        self._text(screen, "AVALIANDO DESEMPENHO", 17, rc.accent_blue, (card.centerx, card.y + 22), anchor="midtop", bold=True)
        episode = int(info.get("episode", 0))
        episodes = int(info.get("episodes", 0)) or 0
        progress = episode / episodes if episodes else 0.0
        bar = pygame.Rect(card.x + 34, card.y + 68, card.width - 68, 10)
        self._round_rect(screen, bar, (28, 39, 53), radius=5)
        if progress > 0:
            fill = pygame.Rect(bar.x, bar.y, max(1, int(bar.width * min(1.0, progress))), bar.height)
            self._round_rect(screen, fill, rc.accent_blue, radius=5)
        label = f"{episode}/{episodes} episódios" if episodes else "iniciando..."
        self._text(screen, label, 10, rc.text, (bar.x, bar.bottom + 10), bold=True)
        self._text(
            screen,
            "Mesmas seeds · sem exploração · sem atualizar a Q-table",
            8,
            rc.text_muted,
            (card.centerx, card.y + 112),
            anchor="midtop",
        )
        self.clock_font  # mantém a fonte carregada
        self._text(screen, "O progresso também aparece no terminal", 8, rc.text_muted, (card.centerx, card.y + 130), anchor="midtop")

    def _draw_comparison_overlay(self, screen: pygame.Surface, result: dict[str, Any]) -> None:
        rc = self.rc
        layer = pygame.Surface((rc.main_width, rc.height), pygame.SRCALPHA)
        layer.fill((5, 8, 13, 185))
        screen.blit(layer, (0, 0))
        card = pygame.Rect(rc.main_width // 2 - 330, rc.height // 2 - 230, 660, 460)
        self._alpha_rect(screen, card, (15, 22, 32), 246, radius=18)
        pygame.draw.rect(screen, (54, 72, 94), card, 1, border_radius=18)
        self._text(screen, "COMPARAÇÃO DE DESEMPENHO", 19, rc.text, (card.centerx, card.y + 22), anchor="midtop", bold=True)
        self._text(
            screen,
            f"{result.get('episodes', 0)} episódios · mesmas sementes · ε=0 na avaliação",
            9,
            rc.text_muted,
            (card.centerx, card.y + 49),
            anchor="midtop",
        )

        traditional = result.get("traditional", {})
        ai = result.get("ai", {})
        rows = [
            ("Tempo médio de espera", "avg_wait", "s"),
            ("Veículos atendidos/ep.", "served_mean", ""),
            ("Fila média", "avg_queue", ""),
            ("Maior espera registrada", "max_wait", "s"),
        ]
        header_y = card.y + 76
        self._text(screen, "MÉTRICA", 8, rc.text_muted, (card.x + 34, header_y), bold=True)
        self._text(screen, "TRADICIONAL", 8, rc.text_muted, (card.x + 350, header_y), anchor="topright", bold=True)
        self._text(screen, "IA", 8, rc.text_muted, (card.x + 486, header_y), anchor="topright", bold=True)
        self._text(screen, "DIFERENÇA", 8, rc.text_muted, (card.right - 34, header_y), anchor="topright", bold=True)
        pygame.draw.line(screen, (39, 53, 72), (card.x + 30, header_y + 17), (card.right - 30, header_y + 17), 1)

        for index, (label, key, unit) in enumerate(rows):
            y = header_y + 30 + index * 42
            trad_value = float(traditional.get(key, 0.0))
            ai_value = float(ai.get(key, 0.0))
            diff = ai_value - trad_value
            better = diff < 0 if key != "served_mean" else diff > 0
            diff_color = rc.green if better and abs(diff) > 1e-9 else (rc.red if abs(diff) > 1e-9 else rc.text_muted)
            self._text(screen, label, 10, rc.text, (card.x + 34, y), bold=True)
            self._text(screen, f"{trad_value:.2f}{unit}", 12, rc.accent_blue, (card.x + 350, y - 2), anchor="topright", bold=True)
            self._text(screen, f"{ai_value:.2f}{unit}", 12, rc.green, (card.x + 486, y - 2), anchor="topright", bold=True)
            self._text(screen, f"{diff:+.2f}{unit}", 12, diff_color, (card.right - 34, y - 2), anchor="topright", bold=True)
            pygame.draw.line(screen, (31, 43, 59), (card.x + 30, y + 30), (card.right - 30, y + 30), 1)

        note = (
            "Avaliação gananciosa, sem exploração e sem atualizar a Q-table. "
            "Valores medidos na simulação."
        )
        self._text(screen, note, 8, rc.text_muted, (card.centerx, card.y + 262), anchor="midtop")
        chart_path = result.get("chart_path")
        report_path = result.get("report_path")
        if chart_path:
            self._text(screen, f"Gráfico: {chart_path}", 8, rc.text_muted, (card.centerx, card.y + 284), anchor="midtop")
        if report_path:
            self._text(screen, f"Relatório: {report_path}", 8, rc.text_muted, (card.centerx, card.y + 300), anchor="midtop")
        if result.get("chart_error"):
            self._text(screen, f"Aviso: {result['chart_error']}", 8, rc.red, (card.centerx, card.y + 318), anchor="midtop")

        self._text(
            screen,
            "Pressione C ou Esc para fechar",
            9,
            rc.yellow,
            (card.centerx, card.bottom - 38),
            anchor="midtop",
            bold=True,
        )


def direction_label(vehicle: Vehicle) -> str:
    return DIRECTION_LABELS.get(vehicle.direction, vehicle.direction)


def axis_of(vehicle: Vehicle) -> str:
    return AXIS_OF[vehicle.direction]


# Mantém os imports de geometria disponíveis para integrações/testes visuais.
DIRECTION_VECTORS = DIRECTION_VECTORS
