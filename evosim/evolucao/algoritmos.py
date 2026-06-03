"""Algoritmos evolutivos que otimizam os pesos do controlador.

Dois algoritmos intercambiáveis sob a mesma interface ``AlgoritmoEvolutivo``:

* ``AlgoritmoGenetico``   — GA clássico: seleção por torneio, elitismo,
  crossover (blend) e mutação gaussiana.
* ``EstrategiaEvolutiva`` — (μ/μ_w, λ)-ES com adaptação de passo (sigma),
  uma versão enxuta no espírito de CMA-ES, robusta para espaços contínuos.

Ambos só mexem no vetor de pesos — os parâmetros do ambiente são intocáveis.
"""
from __future__ import annotations

import abc
import math
from typing import List

from ..mathutils import make_rng
from ..neural.controlador import ControladorNeural
from .genoma import Genoma


class AlgoritmoEvolutivo(abc.ABC):
    def __init__(
        self,
        prototipo: ControladorNeural,
        tamanho_pop: int = 32,
        seed: int = 1234,
    ) -> None:
        self.template = prototipo.to_dict()
        self.n = prototipo.num_pesos()
        self.tamanho_pop = tamanho_pop
        self.rng = make_rng(seed)
        self.geracao = 0
        self.melhor: Genoma | None = None

    def _registrar_melhor(self, pop: List[Genoma]) -> None:
        cand = max(pop, key=lambda g: g.fitness)
        if self.melhor is None or cand.fitness > self.melhor.fitness:
            self.melhor = cand.clone()

    @abc.abstractmethod
    def inicializar(self) -> List[Genoma]: ...

    @abc.abstractmethod
    def proxima_geracao(self, populacao_avaliada: List[Genoma]) -> List[Genoma]: ...


# ---------------------------------------------------------------------------
class AlgoritmoGenetico(AlgoritmoEvolutivo):
    def __init__(
        self,
        prototipo: ControladorNeural,
        tamanho_pop: int = 32,
        seed: int = 1234,
        elite: int = 2,
        taxa_mutacao: float = 0.12,
        forca_mutacao: float = 0.25,
        torneio: int = 3,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__(prototipo, tamanho_pop, seed)
        self.elite = elite
        self.taxa_mutacao = taxa_mutacao
        self.forca_mutacao = forca_mutacao
        self.torneio = torneio
        self.sigma_init = sigma_init

    def inicializar(self) -> List[Genoma]:
        pop = []
        for _ in range(self.tamanho_pop):
            vetor = [self.rng.gauss(0.0, self.sigma_init) for _ in range(self.n)]
            pop.append(Genoma.de_vetor(self.template, vetor, self.geracao))
        return pop

    def _selecionar(self, pop: List[Genoma]) -> Genoma:
        amostra = [self.rng.choice(pop) for _ in range(self.torneio)]
        return max(amostra, key=lambda g: g.fitness)

    def _crossover(self, a: Genoma, b: Genoma) -> List[float]:
        # Crossover uniforme: cada gene vem de um dos pais com prob. 1/2.
        return [x if self.rng.random() < 0.5 else y
                for x, y in zip(a.pesos, b.pesos)]

    def _mutar(self, vetor: List[float]) -> List[float]:
        out = list(vetor)
        for i in range(len(out)):
            if self.rng.random() < self.taxa_mutacao:
                out[i] += self.rng.gauss(0.0, self.forca_mutacao)
        return out

    def proxima_geracao(self, pop: List[Genoma]) -> List[Genoma]:
        self._registrar_melhor(pop)
        ordenada = sorted(pop, key=lambda g: g.fitness, reverse=True)
        self.geracao += 1
        nova: List[Genoma] = [g.clone() for g in ordenada[: self.elite]]
        for g in nova:
            g.geracao = self.geracao
        while len(nova) < self.tamanho_pop:
            pai = self._selecionar(ordenada)
            mae = self._selecionar(ordenada)
            filho_vec = self._mutar(self._crossover(pai, mae))
            nova.append(Genoma.de_vetor(self.template, filho_vec, self.geracao))
        return nova


# ---------------------------------------------------------------------------
class EstrategiaEvolutiva(AlgoritmoEvolutivo):
    """(μ/μ_w, λ)-ES com média ponderada e adaptação de sigma."""

    def __init__(
        self,
        prototipo: ControladorNeural,
        tamanho_pop: int = 32,
        seed: int = 1234,
        sigma: float = 0.3,
        mu: int | None = None,
    ) -> None:
        super().__init__(prototipo, tamanho_pop, seed)
        self.sigma = sigma
        self.mu = mu or max(2, tamanho_pop // 4)
        self.media = [0.0] * self.n
        # pesos de recombinação logarítmicos (estilo CMA-ES).
        ws = [math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)]
        soma = sum(ws)
        self.w = [w / soma for w in ws]

    def _amostrar(self) -> List[float]:
        return [self.media[i] + self.sigma * self.rng.gauss(0.0, 1.0)
                for i in range(self.n)]

    def inicializar(self) -> List[Genoma]:
        return [Genoma.de_vetor(self.template, self._amostrar(), self.geracao)
                for _ in range(self.tamanho_pop)]

    def proxima_geracao(self, pop: List[Genoma]) -> List[Genoma]:
        self._registrar_melhor(pop)
        elites = sorted(pop, key=lambda g: g.fitness, reverse=True)[: self.mu]
        # nova média = combinação ponderada dos melhores.
        nova_media = [0.0] * self.n
        for peso, g in zip(self.w, elites):
            pv = g.pesos
            for i in range(self.n):
                nova_media[i] += peso * pv[i]
        # adaptação simples de sigma pela dispersão dos elites em relação à média.
        desvio = 0.0
        for g in elites:
            pv = g.pesos
            desvio += sum((pv[i] - nova_media[i]) ** 2 for i in range(self.n))
        desvio = math.sqrt(desvio / (self.mu * self.n)) if self.n else self.sigma
        self.sigma = max(1e-3, 0.5 * self.sigma + 0.5 * desvio)
        self.media = nova_media
        self.geracao += 1
        return [Genoma.de_vetor(self.template, self._amostrar(), self.geracao)
                for _ in range(self.tamanho_pop)]
