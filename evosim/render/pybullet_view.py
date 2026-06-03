"""Reprodução visual 3D de um genoma treinado usando o GUI do PyBullet.

Requer ``pip install pybullet``. Carrega um ``Save``, reconstrói a criatura no
``MotorPyBullet`` em modo GUI e roda o controlador em tempo real.

Uso:
    python -m evosim.cli assistir3d --save runs/humanoide.json
"""
from __future__ import annotations

import time

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..fisica.motor_pybullet import MotorPyBullet, disponivel
from ..mathutils import Vec3
from ..persistencia.serializacao import Save


def assistir_save_3d(save: Save, segundos: float = 20.0,
                     ambiente: ConfigAmbiente | None = None) -> None:
    if not disponivel():
        raise RuntimeError("PyBullet não instalado. Rode: pip install pybullet")
    ambiente = ambiente or ConfigAmbiente(**save.ambiente) if save.ambiente else ConfigAmbiente()
    sim = ConfigSimulacao(duracao_segundos=segundos)
    motor = MotorPyBullet(ambiente, gui=True)
    motor.substeps = sim.substeps
    dna = save.dna_obj()
    corpo = motor.construir_criatura(dna, Vec3(0, 0, 0))
    criatura = Criatura(dna, corpo)
    criatura.controlador = save.melhor_genoma().instanciar_controlador()
    for _ in range(sim.passos_totais):
        criatura.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        time.sleep(sim.dt)  # ritmo de tempo real para visualização.
