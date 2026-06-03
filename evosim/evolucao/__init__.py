"""Núcleo evolutivo: genomas e algoritmos."""
from __future__ import annotations

from typing import Callable

from ..neural.controlador import ControladorNeural
from .algoritmos import AlgoritmoEvolutivo, AlgoritmoGenetico, EstrategiaEvolutiva
from .genoma import Genoma


def criar_algoritmo(
    nome: str, tamanho_pop: int = 32, seed: int = 1234, **kw
) -> Callable[[ControladorNeural], AlgoritmoEvolutivo]:
    """Devolve uma factory ``prototipo -> AlgoritmoEvolutivo``."""
    nome = nome.lower()

    def factory(prototipo: ControladorNeural) -> AlgoritmoEvolutivo:
        if nome in ("ga", "genetico"):
            return AlgoritmoGenetico(prototipo, tamanho_pop, seed, **kw)
        if nome in ("es", "estrategia", "cma"):
            return EstrategiaEvolutiva(prototipo, tamanho_pop, seed, **kw)
        raise ValueError(f"Algoritmo desconhecido: {nome}")

    return factory


__all__ = [
    "AlgoritmoEvolutivo",
    "AlgoritmoGenetico",
    "EstrategiaEvolutiva",
    "Genoma",
    "criar_algoritmo",
]
