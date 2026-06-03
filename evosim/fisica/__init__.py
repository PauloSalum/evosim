"""Motores de física (abstração + implementações)."""
from .motor import CorpoCriatura, MotorFisica
from .motor_interno import MotorInterno

__all__ = ["MotorFisica", "CorpoCriatura", "MotorInterno"]
