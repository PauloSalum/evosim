"""Avaliador: roda um indivíduo (ou grupo) com passo de tempo FIXO.

Aqui mora o laço de simulação física integrado ao passo de controle e aos
critérios de parada precoce (early termination), que evitam desperdiçar ciclos
com espécimes inválidos.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..criaturas.dna import CriaturaDNA
from ..fisica.motor import MotorFisica
from ..fisica.motor_interno import MotorInterno
from ..mathutils import Vec3
from ..neural.controlador import ControladorNeural
from .resultado import ResultadoEpisodio

MotorFactory = Callable[[ConfigAmbiente], MotorFisica]


def motor_interno_factory(ambiente: ConfigAmbiente) -> MotorFisica:
    return MotorInterno(ambiente)


class Avaliador:
    def __init__(
        self,
        ambiente: ConfigAmbiente,
        sim: ConfigSimulacao,
        eixo: Vec3 = Vec3(1.0, 0.0, 0.0),
        motor_factory: Optional[MotorFactory] = None,
    ) -> None:
        self.ambiente = ambiente
        self.sim = sim
        self.eixo = eixo.normalized()
        self.motor_factory = motor_factory or motor_interno_factory

    # ------------------------------------------------------------------
    def novo_motor(self) -> MotorFisica:
        motor = self.motor_factory(self.ambiente)
        # repassa o número de subpassos ao motor (interface informal).
        setattr(motor, "substeps", self.sim.substeps)
        return motor

    def dimensoes_controlador(self, dna: CriaturaDNA) -> tuple[int, int]:
        """(num_entradas, num_saidas) que um controlador para este DNA exige."""
        motor = self.novo_motor()
        corpo = motor.construir_criatura(dna, Vec3(0.0, 0.0, 0.0))
        c = Criatura(dna, corpo)
        return c.num_entradas(), c.num_saidas()

    def montar(
        self,
        motor: MotorFisica,
        dna: CriaturaDNA,
        controlador: ControladorNeural,
        base: Vec3,
    ) -> Criatura:
        corpo = motor.construir_criatura(dna, base)
        criatura = Criatura(dna, corpo)
        criatura.controlador = controlador
        return criatura

    # ------------------------------------------------------------------
    # Avaliação de um único indivíduo (modo evolução isolada).
    # ------------------------------------------------------------------
    def avaliar_individuo(
        self,
        dna: CriaturaDNA,
        controlador: ControladorNeural,
        base: Vec3 = Vec3(0.0, 0.0, 0.0),
    ) -> ResultadoEpisodio:
        motor = self.novo_motor()
        criatura = self.montar(motor, dna, controlador, base)
        return self._rodar(motor, criatura)

    def _rodar(self, motor: MotorFisica, criatura: Criatura) -> ResultadoEpisodio:
        sim = self.sim
        dt = sim.dt
        pos0 = criatura.corpo.centro_de_massa()
        res = ResultadoEpisodio()
        passos = sim.passos_totais
        energia_anterior = criatura.corpo.energia_acumulada()

        for passo in range(passos):
            criatura.passo_controle(motor.tempo)
            motor.passo(dt)

            e_atual = criatura.corpo.energia_acumulada()
            criatura.acumular_metricas(e_atual - energia_anterior)
            energia_anterior = e_atual

            cm = criatura.corpo.centro_de_massa()
            tempo = motor.tempo

            # --- critérios de parada precoce ---
            if cm.y < sim.altura_critica:
                res.terminou_cedo, res.motivo = True, "caiu"
                break
            # capotou / deu cambalhota (tronco virado de cabeça para baixo).
            if criatura.corpo.vetor_up_core().y < sim.up_minimo_capotar:
                res.terminou_cedo, res.motivo = True, "capotou"
                break
            # só os pés podem tocar o solo ("se jogar machuca").
            if sim.apenas_pes_no_solo and criatura.corpo.parte_nao_pe_no_solo():
                res.terminou_cedo, res.motivo = True, "contato_indevido"
                break
            if sim.proibe_contato_cabeca and criatura.corpo.parte_proibida_no_solo():
                res.terminou_cedo, res.motivo = True, "cabeca_no_solo"
                break
            if tempo >= sim.janela_aquecimento:
                avanco = (cm - pos0).dot(self.eixo)
                if avanco < sim.deslocamento_minimo:
                    res.terminou_cedo, res.motivo = True, "parado"
                    break

        cm_final = criatura.corpo.centro_de_massa()
        self._preencher(res, criatura, pos0, cm_final, motor.tempo)
        return res

    def _preencher(
        self,
        res: ResultadoEpisodio,
        criatura: Criatura,
        pos0: Vec3,
        cm_final: Vec3,
        tempo: float,
    ) -> None:
        res.deslocamento = cm_final - pos0
        res.distancia_eixo = res.deslocamento.dot(self.eixo)
        res.tempo = max(tempo, 1e-6)
        res.energia = criatura.energia_total
        passos = max(1, criatura.passos)
        res.desvio_vertical_medio = criatura.desvio_vertical_acc / passos
        res.oscilacao_lateral_media = criatura.oscilacao_lateral_acc / passos
