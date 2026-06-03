"""Visualização 3D com matplotlib — FIEL ao treino.

Roda a criatura no **mesmo motor usado no treino** (``MotorInterno``), então o
que você vê em 3D é exatamente o comportamento evoluído. Dois usos:

* ``reproduzir_save``  — assiste/exporta o melhor de um save já treinado.
* ``Monitor3D``        — janela 3D ao vivo que mostra o campeão de cada geração
  ENQUANTO o treino acontece (passada ao ``Executor`` como ``monitor``).

Requer matplotlib (opcional):  pip install matplotlib
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..criaturas.dna import CriaturaDNA
from ..fisica.motor_interno import MotorInterno
from ..mathutils import Vec3
from ..neural.controlador import ControladorNeural
from ..persistencia.serializacao import Save, carregar

Quadro = List[Tuple[Vec3, Vec3, str]]


# ---------------------------------------------------------------------------
# Simulação → quadros (linhas 3D), no motor do treino.
# ---------------------------------------------------------------------------
def simular_quadros(
    dna: CriaturaDNA,
    controlador: ControladorNeural,
    ambiente: ConfigAmbiente,
    sim: ConfigSimulacao,
    cada: int = 3,
) -> List[Quadro]:
    motor = MotorInterno(ambiente)
    motor.substeps = sim.substeps
    cri = Criatura(dna, motor.construir_criatura(dna, Vec3(0, 0, 0)))
    cri.controlador = controlador
    quadros: List[Quadro] = []
    for passo in range(sim.passos_totais):
        cri.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        if passo % cada == 0:
            quadros.append(list(motor.coletar_segmentos_render()))
    return quadros


def _resolver_save(save: Union[str, Save]) -> Save:
    return carregar(save) if isinstance(save, str) else save


def _ambiente_do_save(save: Save, ambiente: Optional[ConfigAmbiente]) -> ConfigAmbiente:
    if ambiente is not None:
        return ambiente
    return ConfigAmbiente(**save.ambiente) if save.ambiente else ConfigAmbiente()


def _importar_plt(headless: bool):
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib não está instalado. Rode:  pip install matplotlib"
        ) from e


def _desenhar(ax, quadro: Quadro, titulo: str) -> None:
    """Desenha um quadro. Eixos: X=avanço, Y=lateral(Z mundo), Z=altura(Y mundo)."""
    ax.clear()
    xs = [v.x for a, b, _ in quadro for v in (a, b)]
    cx = sum(xs) / len(xs) if xs else 0.0
    ax.set_xlim(cx - 2.0, cx + 2.0)   # câmera segue o avanço
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(0.0, 3.0)
    ax.set_xlabel("avanço (X)")
    ax.set_ylabel("lateral")
    ax.set_zlabel("altura")
    ax.set_title(titulo)
    ax.plot([cx - 2, cx + 2], [-2, -2], [0, 0], color="0.75", lw=0.5)
    ax.plot([cx - 2, cx + 2], [2, 2], [0, 0], color="0.75", lw=0.5)
    for a, b, _tag in quadro:
        ax.plot([a.x, b.x], [a.z, b.z], [a.y, b.y], "-", color="#1f77b4", lw=3)


# ---------------------------------------------------------------------------
# Playback de um save já treinado.
# ---------------------------------------------------------------------------
def reproduzir_save(
    save: Union[str, Save],
    ambiente: Optional[ConfigAmbiente] = None,
    sim: Optional[ConfigSimulacao] = None,
    gif: Optional[str] = None,
    fps: int = 30,
    cada: int = 3,
    mostrar: bool = True,
) -> None:
    """Anima o melhor indivíduo de um save em 3D; abre janela e/ou salva GIF."""
    save = _resolver_save(save)
    ambiente = _ambiente_do_save(save, ambiente)
    sim = sim or ConfigSimulacao()
    plt = _importar_plt(headless=bool(gif) and not mostrar)
    from matplotlib import animation

    ctrl = save.melhor_genoma().instanciar_controlador()
    quadros = simular_quadros(save.dna_obj(), ctrl, ambiente, sim, cada=cada)
    if not quadros:
        raise RuntimeError("Nenhum quadro para animar.")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def passo(i):
        _desenhar(ax, quadros[i], f"{save.preset}  —  t={i * cada / 120:.2f}s")
        return []

    anim = animation.FuncAnimation(
        fig, passo, frames=len(quadros), interval=1000.0 / max(1, fps), blit=False
    )
    if gif:
        try:
            anim.save(gif, writer=animation.PillowWriter(fps=fps))
            print(f"   GIF salvo em {gif}")
        except Exception as e:  # pragma: no cover
            print(f"   ⚠ não consegui salvar o GIF ({e}). Tente: pip install pillow")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Monitor AO VIVO durante o treino.
# ---------------------------------------------------------------------------
class Monitor3D:
    """Janela 3D que mostra o campeão de cada geração enquanto o treino roda.

    Passe ``.callback`` como ``monitor`` para ``Executor.evoluir``: a cada
    geração ele reproduz, na mesma janela, o melhor indivíduo até ali.
    """

    def __init__(self, fps: int = 30, cada: int = 4, segundos_max: float = 6.0,
                 cada_geracao: int = 1) -> None:
        self.plt = _importar_plt(headless=False)
        self.plt.ion()
        self.fig = self.plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fps = fps
        self.cada = cada
        self.segundos_max = segundos_max
        self.cada_geracao = max(1, cada_geracao)
        try:
            self.fig.show()
        except Exception:
            pass

    def _clip_sim(self, sim: ConfigSimulacao) -> ConfigSimulacao:
        s = copy.copy(sim)
        s.duracao_segundos = min(sim.duracao_segundos, self.segundos_max)
        return s

    def callback(self, executor, ger: int, stats: dict) -> None:
        if ger % self.cada_geracao != 0:
            return
        genoma = executor.melhor_atual()
        if genoma is None:
            return
        sim = self._clip_sim(executor.sim)
        quadros = simular_quadros(
            executor.dna, genoma.instanciar_controlador(),
            executor.ambiente, sim, cada=self.cada,
        )
        titulo = f"geração {ger}  —  melhor fitness {stats.get('melhor', 0):.2f}"
        for q in quadros:
            if not self.plt.fignum_exists(self.fig.number):
                return  # usuário fechou a janela
            _desenhar(self.ax, q, titulo)
            self.plt.pause(1.0 / max(1, self.fps))

    def fechar(self, manter_aberto: bool = True) -> None:
        self.plt.ioff()
        if manter_aberto and self.plt.fignum_exists(self.fig.number):
            self.plt.show()
