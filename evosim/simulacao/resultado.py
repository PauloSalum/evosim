"""Resultado de um episódio de simulação de um indivíduo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..mathutils import Vec3


@dataclass
class ResultadoEpisodio:
    deslocamento: Vec3 = field(default_factory=Vec3)   # posição final - inicial
    distancia_eixo: float = 0.0                         # avanço no eixo correto
    tempo: float = 0.0                                  # tempo simulado vivido
    energia: float = 0.0                                # torque total gasto
    desvio_vertical_medio: float = 0.0                  # 0 = perfeitamente em pé
    oscilacao_lateral_media: float = 0.0
    terminou_cedo: bool = False
    motivo: str = ""
    fracao_contato_indevido: float = 0.0   # tempo tocando o solo com não-pés
    # competição
    capturas: int = 0
    sobreviveu: bool = False
    distancia_oponente: float = 0.0
