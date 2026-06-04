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
        # Vetor de pesos para "warm-start": continuar a evolução a partir de um
        # indivíduo já treinado (a população inicial nasce ao redor dele).
        self.semente: List[float] | None = None

    def semear(self, vetor: List[float]) -> None:
        """Define o ponto de partida da evolução (continuar de um save)."""
        self.semente = list(vetor)

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
        paciencia: int = 12,
    ) -> None:
        super().__init__(prototipo, tamanho_pop, seed)
        self.elite = elite
        self.taxa_mutacao = taxa_mutacao
        self.forca_mutacao = forca_mutacao
        self.torneio = torneio
        self.sigma_init = sigma_init
        self.paciencia = paciencia
        self._melhor_hist = float("-inf")
        self._estag = 0

    def inicializar(self) -> List[Genoma]:
        pop: List[Genoma] = []
        if self.semente is not None:  # mantém o indivíduo de partida intacto.
            pop.append(Genoma.de_vetor(self.template, self.semente, self.geracao))
        while len(pop) < self.tamanho_pop:
            if self.semente is not None:  # variações ao redor do save.
                vetor = [self.semente[i] + self.rng.gauss(0.0, self.forca_mutacao)
                         for i in range(self.n)]
            else:
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

        # anti-platô: se o melhor não melhora há 'paciencia' gerações, injeta
        # IMIGRANTES aleatórios (diversidade nova) e aumenta a mutação.
        melhor_atual = ordenada[0].fitness
        if melhor_atual > self._melhor_hist + 1e-4:
            self._melhor_hist = melhor_atual
            self._estag = 0
        else:
            self._estag += 1
        forca = self.forca_mutacao * (3.0 if self._estag >= self.paciencia else 1.0)
        n_imigrantes = (self.tamanho_pop // 5) if self._estag >= self.paciencia else 0
        if self._estag >= self.paciencia:
            self._estag = 0

        nova: List[Genoma] = [g.clone() for g in ordenada[: self.elite]]
        for g in nova:
            g.geracao = self.geracao
        for _ in range(n_imigrantes):  # sangue novo para escapar do ótimo local
            vetor = [self.rng.gauss(0.0, self.sigma_init) for _ in range(self.n)]
            nova.append(Genoma.de_vetor(self.template, vetor, self.geracao))
        while len(nova) < self.tamanho_pop:
            pai = self._selecionar(ordenada)
            mae = self._selecionar(ordenada)
            filho_vec = self._crossover(pai, mae)
            for i in range(len(filho_vec)):
                if self.rng.random() < self.taxa_mutacao:
                    filho_vec[i] += self.rng.gauss(0.0, forca)
            nova.append(Genoma.de_vetor(self.template, filho_vec, self.geracao))
        return nova[: self.tamanho_pop]


# ---------------------------------------------------------------------------
class EstrategiaEvolutiva(AlgoritmoEvolutivo):
    """(μ/μ_w, λ)-ES com média ponderada e adaptação de sigma."""

    def __init__(
        self,
        prototipo: ControladorNeural,
        tamanho_pop: int = 32,
        seed: int = 1234,
        sigma: float = 0.4,
        mu: int | None = None,
        sigma_min: float = 0.05,
        sigma_max: float = 0.9,
        paciencia: int = 12,
    ) -> None:
        super().__init__(prototipo, tamanho_pop, seed)
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.paciencia = paciencia       # gerações sem melhora antes de reiniciar
        self._melhor_hist = float("-inf")
        self._estag = 0
        self.mu = mu or max(2, tamanho_pop // 4)
        self.media = [0.0] * self.n
        # pesos de recombinação logarítmicos (estilo CMA-ES).
        ws = [math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)]
        soma = sum(ws)
        self.w = [w / soma for w in ws]

    def semear(self, vetor: List[float]) -> None:
        # Continuar de um save: a busca recomeça centrada no indivíduo treinado.
        super().semear(vetor)
        self.media = list(vetor)

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
        # dispersão dos elites (medida do passo natural da busca).
        desvio = 0.0
        for g in elites:
            pv = g.pesos
            desvio += sum((pv[i] - nova_media[i]) ** 2 for i in range(self.n))
        desvio = math.sqrt(desvio / (self.mu * self.n)) if self.n else self.sigma

        # --- controle de passo ANTI-PLATÔ ---
        # O sigma NÃO pode colapsar para ~0 (era o que estagnava tudo): mantemos
        # um piso e um teto. Além disso, se o melhor histórico não melhora por
        # 'paciencia' gerações, REINICIAMOS a exploração (aumentamos sigma) para
        # escapar do ótimo local — em vez de ficar preso no mesmo padrão.
        melhor_atual = max(g.fitness for g in pop)
        if melhor_atual > self._melhor_hist + 1e-4:
            self._melhor_hist = melhor_atual
            self._estag = 0
        else:
            self._estag += 1

        self.sigma = min(self.sigma_max,
                         max(self.sigma_min, 0.6 * self.sigma + 0.5 * desvio))
        if self._estag >= self.paciencia:        # platô -> reinicia a busca
            self.sigma = min(self.sigma_max, self.sigma * 3.0 + 0.15)
            self._estag = 0

        self.media = nova_media
        self.geracao += 1
        return [Genoma.de_vetor(self.template, self._amostrar(), self.geracao)
                for _ in range(self.tamanho_pop)]
