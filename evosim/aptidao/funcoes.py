"""Funções de aptidão (fitness) selecionáveis.

Cada função recebe um ``ResultadoEpisodio`` e devolve um escalar (maior =
melhor). Novas funções podem ser registradas com ``@registrar_fitness``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

from ..mathutils import clamp

if TYPE_CHECKING:  # evita ciclo aptidao <-> simulacao em tempo de import.
    from ..simulacao.resultado import ResultadoEpisodio

FuncaoAptidao = Callable[["ResultadoEpisodio"], float]

_REGISTRO: Dict[str, FuncaoAptidao] = {}


def registrar_fitness(nome: str):
    def deco(fn: FuncaoAptidao) -> FuncaoAptidao:
        _REGISTRO[nome] = fn
        return fn
    return deco


def obter_fitness(nome: str) -> FuncaoAptidao:
    if nome not in _REGISTRO:
        raise ValueError(f"Fitness desconhecida: {nome}. Opções: {list(_REGISTRO)}")
    return _REGISTRO[nome]


def listar_fitness() -> list:
    return list(_REGISTRO)


# ---------------------------------------------------------------------------
@registrar_fitness("velocidade")
def fitness_velocidade(r: ResultadoEpisodio) -> float:
    """Velocidade máxima: distância no eixo correto / tempo."""
    if r.tempo <= 1e-6:
        return 0.0
    return r.distancia_eixo / r.tempo


@registrar_fitness("eficiencia")
def fitness_eficiencia(r: ResultadoEpisodio) -> float:
    """Eficiência energética: distância / energia gasta."""
    # Exige um mínimo de deslocamento para não premiar criaturas paradas.
    if r.distancia_eixo <= 0.0:
        return 0.0
    return r.distancia_eixo / (r.energia + 1.0)


@registrar_fitness("marcha")
def fitness_marcha(r: "ResultadoEpisodio") -> float:
    """Marcha natural (estilo SCONE): avançar mantendo-se em pé, gastando pouco
    esforço e SEM se arrastar. Em vez de matar o episódio quando algo errado
    acontece (cambalhota, corpo no chão), penaliza no objetivo — assim há
    gradiente para a evolução aprender, e ainda assim esses vícios são punidos.
    """
    avanco = max(0.0, r.distancia_eixo)
    ereto = clamp(1.0 - r.desvio_vertical_medio, 0.0, 1.0)
    esforco_medio = r.energia / (r.tempo + 1.0)
    # progresso é a recompensa principal; sobreviver em pé é secundário; arrastar
    # o corpo e gastar esforço à toa são penalidades leves (não dominam).
    return (2.0 * avanco
            + 0.2 * r.tempo * ereto
            - 0.5 * r.fracao_contato_indevido * r.tempo
            - 0.02 * esforco_medio)


@registrar_fitness("estabilidade")
def fitness_estabilidade(r: ResultadoEpisodio) -> float:
    """Estabilidade: maximiza tempo em pé minimizando desvio e oscilação."""
    base = r.tempo  # recompensa por sobreviver sem cair.
    penalidade = 4.0 * r.desvio_vertical_medio + 2.0 * r.oscilacao_lateral_media
    avanco = 0.2 * r.distancia_eixo  # leve incentivo a andar mantendo equilíbrio.
    return base - penalidade + avanco
