"""Modo Evolução Isolada: uma única espécie evolui por gerações."""
from __future__ import annotations

from typing import Callable, Optional, Union

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.dna import CriaturaDNA
from ..criaturas.presets import criar_preset
from ..evolucao import criar_algoritmo
from ..persistencia.serializacao import Save
from ..simulacao.executor import Executor


def rodar_evolucao_isolada(
    especie: Union[str, CriaturaDNA],
    geracoes: int = 30,
    ambiente: Optional[ConfigAmbiente] = None,
    sim: Optional[ConfigSimulacao] = None,
    fitness: str = "velocidade",
    algoritmo: str = "es",
    tipo_controlador: str = "cpg",
    tamanho_pop: int = 32,
    seed: int = 1234,
    callback: Optional[Callable[[int, dict], None]] = None,
    monitor: Optional[Callable] = None,
) -> Save:
    dna = criar_preset(especie) if isinstance(especie, str) else especie
    ambiente = ambiente or ConfigAmbiente()
    sim = sim or ConfigSimulacao(seed=seed)
    factory = criar_algoritmo(algoritmo, tamanho_pop=tamanho_pop, seed=seed)
    executor = Executor(
        dna, ambiente, sim, factory,
        fitness_nome=fitness, tipo_controlador=tipo_controlador,
    )
    return executor.evoluir(geracoes, callback=callback, monitor=monitor)
