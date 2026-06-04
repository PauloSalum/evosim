"""Avaliação paralela da população em múltiplos núcleos de CPU.

A avaliação de cada indivíduo é **independente** e **determinística** — a
simulação física não usa aleatoriedade (o RNG vive só no algoritmo evolutivo,
no processo principal). Por isso podemos distribuir os indivíduos por vários
processos sem afetar a reprodutibilidade: o resultado é idêntico ao serial.

Cada processo trabalhador é inicializado uma única vez com a morfologia, o
ambiente, a configuração de simulação e a função de fitness (que são fixos
durante todo o treino) e depois recebe apenas o genoma a avaliar.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..aptidao.funcoes import obter_fitness
from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.dna import CriaturaDNA
from ..fisica import criar_motor_factory
from ..mathutils import Vec3
from ..neural.controlador import from_dict as controlador_from_dict
from .avaliador import Avaliador

# Estado por processo trabalhador (preenchido pelo inicializador).
_G: Dict[str, Any] = {}


def _init_worker(
    dna_dict: dict,
    ambiente_dict: dict,
    sim_dict: dict,
    eixo: Tuple[float, float, float],
    fitness_nome: str,
    motor: str = "interno",
) -> None:
    ambiente = ConfigAmbiente(**ambiente_dict)
    sim = ConfigSimulacao(**sim_dict)
    _G["dna"] = CriaturaDNA.from_dict(dna_dict)
    _G["av"] = Avaliador(ambiente, sim, eixo=Vec3(*eixo),
                         motor_factory=criar_motor_factory(motor))
    _G["fit"] = obter_fitness(fitness_nome)


def _avaliar_spec(controlador_spec: dict) -> float:
    """Avalia um genoma (spec do controlador) e devolve seu fitness."""
    ctrl = controlador_from_dict(controlador_spec)
    res = _G["av"].avaliar_individuo(_G["dna"], ctrl)
    return _G["fit"](res)
