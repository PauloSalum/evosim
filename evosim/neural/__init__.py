"""Controladores neurais (cérebros)."""
from __future__ import annotations

from .controlador import ControladorNeural, from_dict
from .cpg import GeradorPadraoCentral
from .mlp import RedeNeuralMLP


def criar_controlador(
    tipo: str, num_entradas: int, num_saidas: int, seed: int = 0, **kw
) -> ControladorNeural:
    if tipo == "mlp":
        return RedeNeuralMLP(num_entradas, num_saidas, seed=seed, **kw)
    if tipo == "cpg":
        return GeradorPadraoCentral(num_entradas, num_saidas, seed=seed, **kw)
    raise ValueError(f"Tipo de controlador desconhecido: {tipo}")


__all__ = [
    "ControladorNeural",
    "RedeNeuralMLP",
    "GeradorPadraoCentral",
    "criar_controlador",
    "from_dict",
]
