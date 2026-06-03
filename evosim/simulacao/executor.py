"""Executor: orquestra o laço de gerações do Modo Evolução Isolada.

Cola tudo: dimensiona o controlador para o DNA, inicializa a população, avalia
cada indivíduo (passo de física + fitness), aplica seleção/mutação via o
``AlgoritmoEvolutivo`` e registra o histórico. Produz um ``Save`` ao final.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional

from ..aptidao.funcoes import obter_fitness
from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.dna import CriaturaDNA
from ..evolucao.algoritmos import AlgoritmoEvolutivo
from ..evolucao.genoma import Genoma
from ..mathutils import Vec3, mean
from ..neural import criar_controlador
from ..neural.controlador import ControladorNeural
from ..persistencia.serializacao import Save
from .avaliador import Avaliador


class Executor:
    def __init__(
        self,
        dna: CriaturaDNA,
        ambiente: ConfigAmbiente,
        sim: ConfigSimulacao,
        algoritmo_factory: Callable[[ControladorNeural], AlgoritmoEvolutivo],
        fitness_nome: str = "velocidade",
        tipo_controlador: str = "cpg",
        eixo: Vec3 = Vec3(1.0, 0.0, 0.0),
    ) -> None:
        self.dna = dna
        self.ambiente = ambiente
        self.sim = sim
        self.fitness_nome = fitness_nome
        self.fitness = obter_fitness(fitness_nome)
        self.avaliador = Avaliador(ambiente, sim, eixo=eixo)
        n_in, n_out = self.avaliador.dimensoes_controlador(dna)
        self.prototipo = criar_controlador(tipo_controlador, n_in, n_out, seed=sim.seed)
        self.algoritmo = algoritmo_factory(self.prototipo)
        self.historico: List[dict] = []
        self.populacao: List[Genoma] = []

    # ------------------------------------------------------------------
    def avaliar_populacao(self, pop: List[Genoma]) -> None:
        for g in pop:
            controlador = g.instanciar_controlador()
            res = self.avaliador.avaliar_individuo(self.dna, controlador)
            g.fitness = self.fitness(res)

    def melhor_atual(self) -> Optional[Genoma]:
        """Melhor genoma da população avaliada atual (para monitores ao vivo)."""
        if not self.populacao:
            return None
        return max(self.populacao, key=lambda g: g.fitness)

    def evoluir(
        self,
        geracoes: int,
        callback: Optional[Callable[[int, dict], None]] = None,
        monitor: Optional[Callable[["Executor", int, dict], None]] = None,
    ) -> Save:
        self.populacao = self.algoritmo.inicializar()
        for ger in range(geracoes):
            self.avaliar_populacao(self.populacao)
            fits = [g.fitness for g in self.populacao]
            stats = {
                "geracao": ger,
                "melhor": max(fits),
                "media": mean(fits),
                "pior": min(fits),
            }
            self.historico.append(stats)
            if callback:
                callback(ger, stats)
            if monitor:  # ex.: Monitor3D — assistir o campeão enquanto treina.
                monitor(self, ger, stats)
            self.populacao = self.algoritmo.proxima_geracao(self.populacao)
        # avalia a última população para registrar o melhor final.
        self.avaliar_populacao(self.populacao)
        self.algoritmo._registrar_melhor(self.populacao)
        return self.para_save()

    # ------------------------------------------------------------------
    def melhores(self, k: int = 3) -> List[Genoma]:
        ordenada = sorted(self.populacao, key=lambda g: g.fitness, reverse=True)
        top = ordenada[:k]
        if self.algoritmo.melhor and (
            not top or self.algoritmo.melhor.fitness >= top[0].fitness
        ):
            top = [self.algoritmo.melhor] + [
                g for g in top if g.id != self.algoritmo.melhor.id
            ]
        return top[:k]

    def para_save(self) -> Save:
        return Save(
            preset=self.dna.preset,
            dna=self.dna.to_dict(),
            ambiente=dataclasses.asdict(self.ambiente),
            fitness_nome=self.fitness_nome,
            geracao=self.algoritmo.geracao,
            historico=self.historico,
            melhores=[g.to_dict() for g in self.melhores()],
        )
