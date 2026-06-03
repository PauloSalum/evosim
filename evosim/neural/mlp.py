"""Rede neural feedforward (MLP) como controlador motor."""
from __future__ import annotations

from typing import List

from ..mathutils import ACTIVATIONS, make_rng, tanh
from .controlador import ControladorNeural, registrar


@registrar
class RedeNeuralMLP(ControladorNeural):
    tipo = "mlp"

    def __init__(
        self,
        num_entradas: int,
        num_saidas: int,
        ocultas: List[int] | None = None,
        ativacao: str = "tanh",
        seed: int = 0,
    ) -> None:
        super().__init__(num_entradas, num_saidas)
        self.ocultas = list(ocultas) if ocultas else [24, 18]
        self.ativacao = ativacao
        self._act = ACTIVATIONS.get(ativacao, tanh)
        # Tamanhos das camadas (inclui entrada e saída).
        self._tamanhos = [num_entradas] + self.ocultas + [num_saidas]
        self._init_pesos(make_rng(seed ^ 0x9E3779B1))

    def _init_pesos(self, rng) -> None:
        self.camadas = []  # cada uma: (W: List[List[float]], b: List[float])
        for nin, nout in zip(self._tamanhos[:-1], self._tamanhos[1:]):
            escala = (1.0 / max(1, nin)) ** 0.5
            W = [[rng.uniform(-escala, escala) for _ in range(nin)] for _ in range(nout)]
            b = [0.0 for _ in range(nout)]
            self.camadas.append((W, b))

    # --- inferência ---------------------------------------------------
    def avaliar(self, entradas: List[float], tempo: float) -> List[float]:
        x = entradas
        ultima = len(self.camadas) - 1
        for li, (W, b) in enumerate(self.camadas):
            y = []
            for r in range(len(W)):
                linha = W[r]
                s = b[r]
                for c in range(len(linha)):
                    s += linha[c] * x[c]
                # saída final sempre em tanh => [-1, 1] (faixa de ativação).
                y.append(tanh(s) if li == ultima else self._act(s))
            x = y
        return x

    # --- vetor de parâmetros ------------------------------------------
    def get_pesos(self) -> List[float]:
        flat: List[float] = []
        for W, b in self.camadas:
            for linha in W:
                flat.extend(linha)
            flat.extend(b)
        return flat

    def set_pesos(self, pesos: List[float]) -> None:
        it = iter(pesos)
        for W, b in self.camadas:
            for linha in W:
                for c in range(len(linha)):
                    linha[c] = next(it)
            for r in range(len(b)):
                b[r] = next(it)

    def num_pesos(self) -> int:
        total = 0
        for nin, nout in zip(self._tamanhos[:-1], self._tamanhos[1:]):
            total += nin * nout + nout
        return total

    # --- serialização -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "num_entradas": self.num_entradas,
            "num_saidas": self.num_saidas,
            "ocultas": self.ocultas,
            "ativacao": self.ativacao,
            "pesos": self.get_pesos(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RedeNeuralMLP":
        obj = cls(
            data["num_entradas"],
            data["num_saidas"],
            ocultas=data.get("ocultas"),
            ativacao=data.get("ativacao", "tanh"),
        )
        obj.set_pesos(data["pesos"])
        return obj
