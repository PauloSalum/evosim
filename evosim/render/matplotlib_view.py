"""Animação 3D leve com matplotlib (opcional: pip install matplotlib).

Grava/exibe os segmentos do motor quadro a quadro. Útil quando não se quer a
dependência nativa do PyBullet, mas se deseja algo mais visual que o ASCII.
"""
from __future__ import annotations

from typing import List

from ..fisica.motor import MotorFisica


def gravar_animacao(quadros: List[list], caminho: str = "evolucao.gif",
                    fps: int = 30) -> None:
    """``quadros``: lista de snapshots de ``motor.coletar_segmentos_render()``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib não instalado: pip install matplotlib") from e

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def desenhar(i):
        ax.clear()
        ax.set_xlim(-2, 8); ax.set_ylim(-2, 2); ax.set_zlim(0, 3)
        for a, b, _ in quadros[i]:
            ax.plot([a.x, b.x], [a.z, b.z], [a.y, b.y], "-", lw=2)
        return []

    anim = animation.FuncAnimation(fig, desenhar, frames=len(quadros), interval=1000 / fps)
    anim.save(caminho, fps=fps)


def coletor(motor: MotorFisica, destino: list, cada: int = 2):
    """Devolve um callback ``on_step`` que acumula quadros em ``destino``."""
    def _cb(m: MotorFisica, passo: int):
        if passo % cada == 0:
            destino.append(list(m.coletar_segmentos_render()))
    return _cb
