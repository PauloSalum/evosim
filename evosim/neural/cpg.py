"""Gerador de Padrão Central (CPG) como controlador motor.

Locomoção biológica é fortemente rítmica. Um CPG produz oscilações acopladas
naturais para cada junta, moduladas por realimentação sensorial. Aqui cada
saída é um oscilador senoidal (frequência, amplitude, fase, offset) somado a
uma projeção linear dos sensores — tudo codificado no vetor de parâmetros que
a evolução otimiza. CPGs costumam aprender marchas estáveis mais rápido que
MLPs puras.
"""
from __future__ import annotations

import math
from typing import List

from ..mathutils import make_rng, tanh
from .controlador import ControladorNeural, registrar

_2PI = 2.0 * math.pi


@registrar
class GeradorPadraoCentral(ControladorNeural):
    tipo = "cpg"

    def __init__(
        self,
        num_entradas: int,
        num_saidas: int,
        freq_base: float = 1.4,
        seed: int = 0,
    ) -> None:
        super().__init__(num_entradas, num_saidas)
        self.freq_base = freq_base
        rng = make_rng(seed ^ 0x85EBCA6B)
        # Por saída: [freq, amplitude, fase, offset].
        self.osc = [
            [rng.uniform(0.5, 1.5), rng.uniform(0.2, 0.8),
             rng.uniform(0.0, _2PI), rng.uniform(-0.2, 0.2)]
            for _ in range(num_saidas)
        ]
        # Matriz de realimentação sensorial (num_saidas x num_entradas).
        self.fb = [[rng.uniform(-0.3, 0.3) for _ in range(num_entradas)]
                   for _ in range(num_saidas)]

    def avaliar(self, entradas: List[float], tempo: float) -> List[float]:
        saidas = []
        for i in range(self.num_saidas):
            freq, amp, fase, off = self.osc[i]
            ritmo = amp * math.sin(_2PI * (self.freq_base * freq) * tempo + fase)
            linha = self.fb[i]
            s = 0.0
            for c in range(len(linha)):
                s += linha[c] * entradas[c]
            saidas.append(tanh(ritmo + off + s))
        return saidas

    # --- vetor de parâmetros ------------------------------------------
    def get_pesos(self) -> List[float]:
        flat: List[float] = []
        for o in self.osc:
            flat.extend(o)
        for linha in self.fb:
            flat.extend(linha)
        return flat

    def set_pesos(self, pesos: List[float]) -> None:
        it = iter(pesos)
        for o in self.osc:
            for k in range(4):
                o[k] = next(it)
        for linha in self.fb:
            for c in range(len(linha)):
                linha[c] = next(it)

    def num_pesos(self) -> int:
        return self.num_saidas * 4 + self.num_saidas * self.num_entradas

    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "num_entradas": self.num_entradas,
            "num_saidas": self.num_saidas,
            "freq_base": self.freq_base,
            "pesos": self.get_pesos(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeradorPadraoCentral":
        obj = cls(data["num_entradas"], data["num_saidas"],
                  freq_base=data.get("freq_base", 1.4))
        obj.set_pesos(data["pesos"])
        return obj
