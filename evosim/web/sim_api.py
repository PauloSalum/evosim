"""Ponte simulação → frames JSON para o frontend web.

Roda a criatura no ``MotorInterno`` (o mesmo motor do treino) e devolve os
segmentos quadro a quadro como listas de floats, prontos para o visualizador
3D do navegador. Permite sobrepor parâmetros do ambiente (ex.: gravidade) em
tempo real sem retreinar.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..criaturas.dna import CriaturaDNA
from ..fisica.motor_interno import MotorInterno
from ..mathutils import Vec3
from ..neural.controlador import ControladorNeural
from ..persistencia.serializacao import Save

# Cada frame: lista de segmentos; cada segmento = [ax, ay, az, bx, by, bz].
Frame = List[List[float]]


def simular_frames(
    dna: CriaturaDNA,
    controlador: ControladorNeural,
    ambiente: ConfigAmbiente,
    sim: ConfigSimulacao,
    cada: int = 2,
    segundos: Optional[float] = None,
) -> List[Frame]:
    motor = MotorInterno(ambiente)
    motor.substeps = sim.substeps
    cri = Criatura(dna, motor.construir_criatura(dna, Vec3(0, 0, 0)))
    cri.controlador = controlador
    passos = sim.passos_totais if segundos is None else int(segundos / sim.dt)
    frames: List[Frame] = []
    for p in range(passos):
        cri.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        if p % cada == 0:
            frames.append([
                [a.x, a.y, a.z, b.x, b.y, b.z]
                for a, b, _ in motor.coletar_segmentos_render()
            ])
    return frames


def ambiente_com_override(base: dict, override: Optional[dict]) -> ConfigAmbiente:
    dados = dict(base) if base else {}
    if override:
        dados.update({k: v for k, v in override.items() if v is not None})
    # remove chaves desconhecidas para não quebrar o dataclass.
    validos = ConfigAmbiente().__dict__.keys()
    dados = {k: v for k, v in dados.items() if k in validos}
    return ConfigAmbiente(**dados)


def frames_de_save(
    save: Save,
    override_ambiente: Optional[dict] = None,
    segundos: float = 8.0,
    cada: int = 2,
) -> dict:
    """Simula o melhor indivíduo de um save e devolve frames + metadados."""
    ambiente = ambiente_com_override(save.ambiente, override_ambiente)
    sim = ConfigSimulacao(duracao_segundos=segundos)
    dna = save.dna_obj()
    ctrl = save.melhor_genoma().instanciar_controlador()
    frames = simular_frames(dna, ctrl, ambiente, sim, cada=cada, segundos=segundos)
    return {
        "preset": save.preset,
        "dt": sim.dt * cada,
        "frames": frames,
    }
