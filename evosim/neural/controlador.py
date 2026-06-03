"""Interface dos controladores motores ("cérebros").

Todo controlador expõe um **vetor de parâmetros plano** (``get_pesos`` /
``set_pesos``). É sobre esse vetor que o algoritmo evolutivo opera — ele não
precisa saber se por baixo há uma MLP ou um gerador de padrão central.
"""
from __future__ import annotations

import abc
from typing import Dict, List, Type


class ControladorNeural(abc.ABC):
    tipo: str = "abstrato"

    def __init__(self, num_entradas: int, num_saidas: int) -> None:
        self.num_entradas = num_entradas
        self.num_saidas = num_saidas

    @abc.abstractmethod
    def avaliar(self, entradas: List[float], tempo: float) -> List[float]:
        """Mapeia sensores (+ tempo) em ativações musculares em [-1, 1]."""

    @abc.abstractmethod
    def get_pesos(self) -> List[float]: ...

    @abc.abstractmethod
    def set_pesos(self, pesos: List[float]) -> None: ...

    @abc.abstractmethod
    def num_pesos(self) -> int: ...

    @abc.abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: dict) -> "ControladorNeural": ...

    def clone(self) -> "ControladorNeural":
        return from_dict(self.to_dict())


# Registro de tipos para (de)serialização polimórfica.
_REGISTRO: Dict[str, Type[ControladorNeural]] = {}


def registrar(cls: Type[ControladorNeural]) -> Type[ControladorNeural]:
    _REGISTRO[cls.tipo] = cls
    return cls


def from_dict(data: dict) -> ControladorNeural:
    tipo = data["tipo"]
    if tipo not in _REGISTRO:
        raise ValueError(f"Controlador desconhecido: {tipo}")
    return _REGISTRO[tipo].from_dict(data)
