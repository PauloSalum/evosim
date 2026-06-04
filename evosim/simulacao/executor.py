"""Executor: orquestra o laço de gerações do Modo Evolução Isolada.

Cola tudo: dimensiona o controlador para o DNA, inicializa a população, avalia
cada indivíduo (passo de física + fitness), aplica seleção/mutação via o
``AlgoritmoEvolutivo`` e registra o histórico. Produz um ``Save`` ao final.
"""
from __future__ import annotations

import dataclasses
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional

from ..aptidao.funcoes import obter_fitness
from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.dna import CriaturaDNA
from ..evolucao.algoritmos import AlgoritmoEvolutivo
from ..evolucao.genoma import Genoma
from ..fisica import criar_motor_factory
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
        n_workers: int = 1,
        genoma_inicial: Optional[Genoma] = None,
        motor: str = "interno",
    ) -> None:
        self.dna = dna
        self.ambiente = ambiente
        self.sim = sim
        self.fitness_nome = fitness_nome
        self.fitness = obter_fitness(fitness_nome)
        self.motor_nome = motor
        self.avaliador = Avaliador(ambiente, sim, eixo=eixo,
                                   motor_factory=criar_motor_factory(motor))
        if genoma_inicial is not None:
            # Continuar de um save: usa a MESMA arquitetura/pesos como protótipo
            # e semeia a busca ao redor desse indivíduo.
            self.prototipo = genoma_inicial.instanciar_controlador()
        else:
            n_in, n_out = self.avaliador.dimensoes_controlador(dna)
            self.prototipo = criar_controlador(tipo_controlador, n_in, n_out, seed=sim.seed)
        self.algoritmo = algoritmo_factory(self.prototipo)
        if genoma_inicial is not None:
            self.algoritmo.semear(genoma_inicial.pesos)
        self.historico: List[dict] = []
        self.populacao: List[Genoma] = []
        # n_workers<=0 => usa todos os núcleos. A avaliação é determinística e
        # independente, então o resultado é idêntico ao serial.
        self.n_workers = n_workers if (n_workers and n_workers > 0) else (os.cpu_count() or 1)
        self._pool: Optional[ProcessPoolExecutor] = None

    # ------------------------------------------------------------------
    def _garantir_pool(self) -> Optional[ProcessPoolExecutor]:
        if self.n_workers <= 1:
            return None
        if self._pool is None:
            from .paralelo import _init_worker
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_init_worker,
                initargs=(
                    self.dna.to_dict(),
                    dataclasses.asdict(self.ambiente),
                    dataclasses.asdict(self.sim),
                    self.avaliador.eixo.as_tuple(),
                    self.fitness_nome,
                    self.motor_nome,
                ),
            )
        return self._pool

    def _fechar_pool(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self.avaliador.fechar()

    def avaliar_populacao(self, pop: List[Genoma]) -> None:
        pool = self._garantir_pool()
        if pool is None:  # serial
            for g in pop:
                res = self.avaliador.avaliar_individuo(self.dna, g.instanciar_controlador())
                g.fitness = self.fitness(res)
            return
        from .paralelo import _avaliar_spec
        specs = [g.controlador_spec for g in pop]
        for g, fit in zip(pop, pool.map(_avaliar_spec, specs)):
            g.fitness = fit

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
        # geracoes <= 0 => ilimitado: roda até um parar cooperativo (o monitor
        # levanta uma exceção) ou KeyboardInterrupt.
        ger_iter = itertools.count() if geracoes <= 0 else range(geracoes)
        try:
            self.populacao = self.algoritmo.inicializar()
            for ger in ger_iter:
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
        finally:
            self._fechar_pool()

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
