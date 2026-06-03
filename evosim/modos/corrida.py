"""Modo Corrida (Sandbox): vários espécimes competem lado a lado.

Carrega criaturas de *saves diferentes* (treinadas de forma independente) e as
coloca em raias paralelas numa pista reta. Mede o avanço de cada uma no eixo da
pista. É a base para comparar gerações/espécies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..criaturas.dna import CriaturaDNA
from ..fisica.motor_interno import MotorInterno
from ..mathutils import Vec3
from ..neural.controlador import ControladorNeural
from ..persistencia.serializacao import Save


@dataclass
class Competidor:
    nome: str
    dna: CriaturaDNA
    controlador: ControladorNeural

    @classmethod
    def de_save(cls, save: Save, nome: Optional[str] = None) -> "Competidor":
        genoma = save.melhor_genoma()
        return cls(
            nome=nome or f"{save.preset}#g{save.geracao}",
            dna=save.dna_obj(),
            controlador=genoma.instanciar_controlador(),
        )


def rodar_corrida(
    competidores: List[Competidor],
    ambiente: Optional[ConfigAmbiente] = None,
    sim: Optional[ConfigSimulacao] = None,
    eixo: Vec3 = Vec3(1.0, 0.0, 0.0),
    espacamento: float = 1.5,
    on_step=None,
) -> List[Tuple[str, float]]:
    """Roda a corrida e devolve ranking [(nome, distancia)] decrescente."""
    ambiente = ambiente or ConfigAmbiente()
    sim = sim or ConfigSimulacao()
    motor = MotorInterno(ambiente)
    setattr(motor, "substeps", sim.substeps)
    eixo = eixo.normalized()

    pistas = []
    n = len(competidores)
    for i, comp in enumerate(competidores):
        z = (i - (n - 1) / 2.0) * espacamento
        corpo = motor.construir_criatura(comp.dna, Vec3(0.0, 0.0, z))
        cri = Criatura(comp.dna, corpo)
        cri.controlador = comp.controlador
        pistas.append((comp.nome, cri, corpo.centro_de_massa()))

    for passo in range(sim.passos_totais):
        for _nome, cri, _p0 in pistas:
            cri.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        if on_step is not None:
            on_step(motor, passo)

    ranking = [
        (nome, (cri.corpo.centro_de_massa() - p0).dot(eixo))
        for nome, cri, p0 in pistas
    ]
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking
