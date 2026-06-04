"""Motores de física (abstração + implementações + seleção)."""
from typing import Callable

from ..config import ConfigAmbiente
from .motor import CorpoCriatura, MotorFisica
from .motor_interno import MotorInterno

MotorFactory = Callable[[ConfigAmbiente], MotorFisica]


def pybullet_disponivel() -> bool:
    from .motor_pybullet import disponivel
    return disponivel()


def criar_motor_factory(nome: str = "auto") -> MotorFactory:
    """Devolve uma fábrica ``ambiente -> MotorFisica``.

    * ``interno``  — solver determinístico em Python puro (sem dependências).
    * ``pybullet`` — corpos rígidos articulados (física natural; requer pybullet).
    * ``auto``     — pybullet se instalado, senão interno.
    """
    nome = (nome or "auto").lower()
    usar_pb = nome in ("pybullet", "bullet") or (nome == "auto" and pybullet_disponivel())
    if usar_pb:
        from .motor_pybullet import MotorPyBullet
        return lambda amb: MotorPyBullet(amb)
    return lambda amb: MotorInterno(amb)


__all__ = [
    "MotorFisica", "CorpoCriatura", "MotorInterno",
    "criar_motor_factory", "pybullet_disponivel", "MotorFactory",
]
