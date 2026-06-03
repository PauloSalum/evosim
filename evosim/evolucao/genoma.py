"""Genoma: a unidade que a evolução manipula.

Um genoma carrega a especificação completa do controlador (tipo + topologia +
vetor de pesos) e, opcionalmente, genes morfológicos. A evolução opera sobre o
**vetor de pesos**; a morfologia só muda se explicitamente habilitada, e ainda
assim dentro dos limites definidos pelo preset.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import List, Optional

from ..neural import controlador as ctrl_mod
from ..neural.controlador import ControladorNeural

_contador = itertools.count(1)


@dataclass
class Genoma:
    controlador_spec: dict
    fitness: float = float("-inf")
    id: int = field(default_factory=lambda: next(_contador))
    geracao: int = 0
    genes_morfologia: Optional[dict] = None

    # --- pesos --------------------------------------------------------
    @property
    def pesos(self) -> List[float]:
        return self.controlador_spec["pesos"]

    @pesos.setter
    def pesos(self, v: List[float]) -> None:
        self.controlador_spec["pesos"] = list(v)

    def num_pesos(self) -> int:
        return len(self.controlador_spec["pesos"])

    # --- instanciação -------------------------------------------------
    def instanciar_controlador(self) -> ControladorNeural:
        return ctrl_mod.from_dict(self.controlador_spec)

    # --- utilidades ---------------------------------------------------
    @classmethod
    def de_vetor(cls, template: dict, vetor: List[float], geracao: int = 0) -> "Genoma":
        spec = copy.deepcopy(template)
        spec["pesos"] = list(vetor)
        return cls(controlador_spec=spec, geracao=geracao)

    def clone(self) -> "Genoma":
        return Genoma(
            controlador_spec=copy.deepcopy(self.controlador_spec),
            fitness=self.fitness,
            geracao=self.geracao,
            genes_morfologia=copy.deepcopy(self.genes_morfologia),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "geracao": self.geracao,
            "controlador": self.controlador_spec,
            "genes_morfologia": self.genes_morfologia,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genoma":
        return cls(
            controlador_spec=data["controlador"],
            fitness=data.get("fitness", float("-inf")),
            geracao=data.get("geracao", 0),
            genes_morfologia=data.get("genes_morfologia"),
        )
