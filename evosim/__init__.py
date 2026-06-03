"""EvoSim — simulação de evolução de caminhada e corrida biológica em 3D.

Sistema modular para evoluir, comparar e co-evoluir criaturas biomecânicas.

Camadas principais
------------------
* ``evosim.fisica``      — motor de física abstrato (``MotorFisica``) com
  implementações ``MotorInterno`` (determinístico, sem deps) e ``MotorPyBullet``.
* ``evosim.criaturas``   — DNA morfológico, músculos, presets e a ``Criatura``.
* ``evosim.neural``      — controladores (``RedeNeuralMLP``, ``GeradorPadraoCentral``).
* ``evosim.aptidao``     — funções de fitness selecionáveis.
* ``evosim.evolucao``    — ``Genoma`` e algoritmos (GA / Estratégia Evolutiva).
* ``evosim.simulacao``   — ``Avaliador`` (passo fixo + parada precoce) e ``Executor``.
* ``evosim.modos``       — evolução isolada, corrida (sandbox), caça-caçador.
* ``evosim.persistencia``— save/load de genomas e histórico.
* ``evosim.render``      — visualização ASCII / matplotlib / PyBullet.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import ConfigAmbiente, ConfigSimulacao

__all__ = ["ConfigAmbiente", "ConfigSimulacao", "__version__"]
