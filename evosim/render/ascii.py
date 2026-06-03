"""Renderizador ASCII — assista à evolução em qualquer terminal, sem GUI.

Projeta os segmentos 3D do motor num plano (lateral X-Y por padrão) e desenha
um quadro de texto. É deliberadamente simples e sem dependências; para
visualização 3D real use ``render.pybullet_view`` ou ``render.matplotlib_view``.
"""
from __future__ import annotations

import sys
import time
from typing import List, Tuple

from ..fisica.motor import MotorFisica
from ..mathutils import Vec3


def _proj(p: Vec3, plano: str) -> Tuple[float, float]:
    if plano == "xz":  # vista de cima
        return p.x, p.z
    if plano == "zy":  # vista de frente
        return p.z, p.y
    return p.x, p.y    # vista lateral (padrão)


def renderizar_quadro(
    motor: MotorFisica, largura: int = 78, altura: int = 22, plano: str = "xy",
    janela: float = 6.0,
) -> str:
    segs = motor.coletar_segmentos_render()
    grade = [[" "] * largura for _ in range(altura)]

    # linha do chão na altura y=0 (apenas para o plano lateral).
    pts: List[Tuple[float, float]] = []
    for a, b, _ in segs:
        pts.append(_proj(a, plano))
        pts.append(_proj(b, plano))
    if not pts:
        return "\n".join("".join(l) for l in grade)

    cx = sum(p[0] for p in pts) / len(pts)
    minv = min(p[1] for p in pts)
    # câmera segue o centro horizontal; eixo vertical ancorado no chão.
    x0, x1 = cx - janela, cx + janela
    y0, y1 = min(minv, 0.0) - 0.2, min(minv, 0.0) + janela

    def para_grade(x: float, y: float) -> Tuple[int, int]:
        gx = int((x - x0) / (x1 - x0) * (largura - 1))
        gy = int((y - y0) / (y1 - y0) * (altura - 1))
        return gx, (altura - 1 - gy)

    # solo
    for gx in range(largura):
        _, gy = para_grade(0.0, 0.0)
        if 0 <= gy < altura:
            grade[gy][gx] = "."

    for a, b, _ in segs:
        ax, ay = _proj(a, plano)
        bx, by = _proj(b, plano)
        gx0, gy0 = para_grade(ax, ay)
        gx1, gy1 = para_grade(bx, by)
        for (gx, gy) in _linha(gx0, gy0, gx1, gy1):
            if 0 <= gx < largura and 0 <= gy < altura:
                grade[gy][gx] = "#"
    return "\n".join("".join(l) for l in grade)


def _linha(x0, y0, x1, y1):
    """Bresenham."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def assistir_terminal(motor: MotorFisica, passo: int, cada: int = 6,
                      plano: str = "xy", fps: float = 30.0) -> None:
    """Callback ``on_step`` para corrida/caça: imprime quadros no terminal."""
    if passo % cada != 0:
        return
    quadro = renderizar_quadro(motor, plano=plano)
    sys.stdout.write("\x1b[H\x1b[2J")  # limpa a tela (ANSI)
    sys.stdout.write(quadro + f"\n  t={motor.tempo:5.2f}s\n")
    sys.stdout.flush()
    if fps > 0:
        time.sleep(1.0 / fps)
