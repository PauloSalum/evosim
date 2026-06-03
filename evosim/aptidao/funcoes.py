"""Funções de aptidão (fitness) selecionáveis.

Cada função recebe um ``ResultadoEpisodio`` e devolve um escalar (maior =
melhor). Novas funções podem ser registradas com ``@registrar_fitness``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

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


@registrar_fitness("estabilidade")
def fitness_estabilidade(r: ResultadoEpisodio) -> float:
    """Estabilidade: maximiza tempo em pé minimizando desvio e oscilação."""
    base = r.tempo  # recompensa por sobreviver sem cair.
    penalidade = 4.0 * r.desvio_vertical_medio + 2.0 * r.oscilacao_lateral_media
    avanco = 0.2 * r.distancia_eixo  # leve incentivo a andar mantendo equilíbrio.
    return base - penalidade + avanco
