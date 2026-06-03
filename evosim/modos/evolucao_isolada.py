"""Modo Evolução Isolada: uma única espécie evolui por gerações."""
from __future__ import annotations

from typing import Callable, Optional, Union

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.dna import CriaturaDNA
from ..criaturas.presets import criar_preset
from ..evolucao import criar_algoritmo
from ..evolucao.genoma import Genoma
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
    n_workers: int = 1,
    genoma_inicial: Optional[Genoma] = None,
) -> Save:
    dna = criar_preset(especie) if isinstance(especie, str) else especie
    ambiente = ambiente or ConfigAmbiente()
    sim = sim or ConfigSimulacao(seed=seed)
    factory = criar_algoritmo(algoritmo, tamanho_pop=tamanho_pop, seed=seed)
    executor = Executor(
        dna, ambiente, sim, factory,
        fitness_nome=fitness, tipo_controlador=tipo_controlador,
        n_workers=n_workers, genoma_inicial=genoma_inicial,
    )
    return executor.evoluir(geracoes, callback=callback, monitor=monitor)


def continuar_evolucao(
    save: Save,
    geracoes: int = 30,
    ambiente: Optional[ConfigAmbiente] = None,
    sim: Optional[ConfigSimulacao] = None,
    fitness: Optional[str] = None,
    algoritmo: str = "es",
    tamanho_pop: int = 32,
    seed: int = 1234,
    callback: Optional[Callable[[int, dict], None]] = None,
    monitor: Optional[Callable] = None,
    n_workers: int = 1,
) -> Save:
    """Continua a evolução a partir de um save (warm-start).

    Reaproveita a morfologia e os pesos do melhor indivíduo, recomeçando a
    busca ao redor dele. Pode trocar o ambiente (ex.: outra gravidade), a
    fitness e o número de núcleos.
    """
    dna = save.dna_obj()
    genoma = save.melhor_genoma()
    ambiente = ambiente or (ConfigAmbiente(**save.ambiente) if save.ambiente
                            else ConfigAmbiente())
    sim = sim or ConfigSimulacao(seed=seed)
    return rodar_evolucao_isolada(
        dna, geracoes=geracoes, ambiente=ambiente, sim=sim,
        fitness=fitness or save.fitness_nome, algoritmo=algoritmo,
        tamanho_pop=tamanho_pop, seed=seed, callback=callback, monitor=monitor,
        n_workers=n_workers, genoma_inicial=genoma,
    )
