"""Modo Caça e Caçador: co-evolução de duas espécies na mesma arena.

O Caçador é recompensado por encostar na Presa; a Presa, por sobreviver/manter
distância. A cada passo, cada criatura recebe a posição relativa da outra como
alvo dinâmico (via ``definir_alvo``), que entra no vetor de sensores. As duas
populações evoluem em paralelo contra o campeão atual da espécie rival (estilo
Hall of Fame), produzindo uma corrida armamentista.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..criaturas.dna import CriaturaDNA
from ..evolucao.algoritmos import AlgoritmoEvolutivo
from ..evolucao.genoma import Genoma
from ..fisica.motor_interno import MotorInterno
from ..mathutils import Vec3, mean
from ..neural import criar_controlador
from ..neural.controlador import ControladorNeural
from ..persistencia.serializacao import Save
from ..simulacao.avaliador import Avaliador


@dataclass
class ResultadoCaca:
    capturou: bool
    tempo: float
    distancia_final: float


def rodar_episodio(
    dna_cacador: CriaturaDNA, ctrl_cacador: ControladorNeural,
    dna_presa: CriaturaDNA, ctrl_presa: ControladorNeural,
    ambiente: ConfigAmbiente, sim: ConfigSimulacao,
    raio_captura: float = 0.5, separacao: float = 3.0,
    on_step=None,
) -> ResultadoCaca:
    motor = MotorInterno(ambiente)
    setattr(motor, "substeps", sim.substeps)

    corpo_c = motor.construir_criatura(dna_cacador, Vec3(0.0, 0.0, 0.0))
    corpo_p = motor.construir_criatura(dna_presa, Vec3(separacao, 0.0, 0.0))
    cacador = Criatura(dna_cacador, corpo_c); cacador.controlador = ctrl_cacador
    presa = Criatura(dna_presa, corpo_p); presa.controlador = ctrl_presa

    capturou = False
    dist = separacao
    for passo in range(sim.passos_totais):
        pc = corpo_c.pos_core()
        pp = corpo_p.pos_core()
        corpo_c.definir_alvo(pp)   # caçador persegue a presa
        corpo_p.definir_alvo(pc)   # presa foge do caçador (sensor de ameaça)
        cacador.passo_controle(motor.tempo)
        presa.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        if on_step is not None:
            on_step(motor, passo)
        dist = corpo_c.pos_core().distance(corpo_p.pos_core())
        if dist <= raio_captura:
            capturou = True
            break
    return ResultadoCaca(capturou, motor.tempo, dist)


def fitness_cacador(r: ResultadoCaca, duracao: float) -> float:
    if r.capturou:
        return 10.0 + (duracao - r.tempo)      # capturar rápido vale mais.
    return -r.distancia_final                   # quanto mais perto, melhor.


def fitness_presa(r: ResultadoCaca, duracao: float) -> float:
    if r.capturou:
        return r.tempo                          # sobreviver mais tempo é melhor.
    return duracao + r.distancia_final          # sobreviveu e manteve distância.


class CoEvolucao:
    def __init__(
        self,
        dna_cacador: CriaturaDNA,
        dna_presa: CriaturaDNA,
        ambiente: ConfigAmbiente,
        sim: ConfigSimulacao,
        algo_factory: Callable[[ControladorNeural], AlgoritmoEvolutivo],
        tipo_controlador: str = "cpg",
        raio_captura: float = 0.5,
    ) -> None:
        self.dna_c, self.dna_p = dna_cacador, dna_presa
        self.ambiente, self.sim = ambiente, sim
        self.raio_captura = raio_captura
        av = Avaliador(ambiente, sim)
        nin_c, nout_c = av.dimensoes_controlador(dna_cacador)
        nin_p, nout_p = av.dimensoes_controlador(dna_presa)
        self.proto_c = criar_controlador(tipo_controlador, nin_c, nout_c, seed=sim.seed)
        self.proto_p = criar_controlador(tipo_controlador, nin_p, nout_p, seed=sim.seed + 1)
        self.algo_c = algo_factory(self.proto_c)
        self.algo_p = algo_factory(self.proto_p)
        self.historico: List[dict] = []

    def _episodio(self, gc: Genoma, gp: Genoma) -> ResultadoCaca:
        return rodar_episodio(
            self.dna_c, gc.instanciar_controlador(),
            self.dna_p, gp.instanciar_controlador(),
            self.ambiente, self.sim, raio_captura=self.raio_captura,
        )

    def evoluir(self, geracoes: int, callback=None) -> Tuple[Save, Save]:
        pop_c = self.algo_c.inicializar()
        pop_p = self.algo_p.inicializar()
        champ_c, champ_p = pop_c[0], pop_p[0]
        dur = self.sim.duracao_segundos
        for ger in range(geracoes):
            # caçadores contra o campeão presa atual.
            for g in pop_c:
                g.fitness = fitness_cacador(self._episodio(g, champ_p), dur)
            champ_c = max(pop_c, key=lambda g: g.fitness)
            # presas contra o campeão caçador atual.
            for g in pop_p:
                g.fitness = fitness_presa(self._episodio(champ_c, g), dur)
            champ_p = max(pop_p, key=lambda g: g.fitness)

            stats = {
                "geracao": ger,
                "cacador_melhor": champ_c.fitness,
                "presa_melhor": champ_p.fitness,
                "cacador_media": mean(g.fitness for g in pop_c),
                "presa_media": mean(g.fitness for g in pop_p),
            }
            self.historico.append(stats)
            if callback:
                callback(ger, stats)
            pop_c = self.algo_c.proxima_geracao(pop_c)
            pop_p = self.algo_p.proxima_geracao(pop_p)

        save_c = self._save(self.dna_c, self.algo_c, champ_c, "cacador")
        save_p = self._save(self.dna_p, self.algo_p, champ_p, "presa")
        return save_c, save_p

    def _save(self, dna, algo, champ, papel) -> Save:
        import dataclasses
        melhor = algo.melhor or champ
        return Save(
            preset=dna.preset,
            dna=dna.to_dict(),
            ambiente=dataclasses.asdict(self.ambiente),
            fitness_nome=papel,
            geracao=algo.geracao,
            historico=self.historico,
            melhores=[melhor.to_dict(), champ.to_dict()],
        )
