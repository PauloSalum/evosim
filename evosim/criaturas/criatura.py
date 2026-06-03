"""A classe `Criatura`: une morfologia, corpo físico, músculos e cérebro.

Responsável pelo laço sensor → controlador → músculo → atuador a cada passo
de simulação. Toda leitura de sensor e composição do vetor de entrada da rede
acontece aqui, em ordem **fixa e determinística**.
"""
from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

from ..mathutils import Vec3, clamp
from .dna import CriaturaDNA, SegmentoDNA
from .musculo import Musculo

if TYPE_CHECKING:
    from ..fisica.motor import CorpoCriatura
    from ..neural.controlador import ControladorNeural

# Escalas para normalizar sensores em faixas próximas de [-1, 1].
_ESCALA_VEL = 0.1
_ESCALA_DIST = 0.05


class Criatura:
    def __init__(
        self,
        dna: CriaturaDNA,
        corpo: "CorpoCriatura",
        suavizacao_muscular: float = 0.18,
    ) -> None:
        self.dna = dna
        self.corpo = corpo
        self.musculos: List[Musculo] = [
            Musculo(i, suavizacao_muscular) for i in range(corpo.num_juntas())
        ]
        self.controlador: Optional["ControladorNeural"] = None
        # contagem de segmentos-pé para dimensionar sensores.
        self._n_pes = sum(1 for s in dna.segmentos if s.eh_pe)
        # acumuladores de métricas de fitness.
        self.energia_total: float = 0.0
        self.desvio_vertical_acc: float = 0.0
        self.oscilacao_lateral_acc: float = 0.0
        self.passos: int = 0
        self.pos_inicial: Vec3 = corpo.centro_de_massa()

    # ------------------------------------------------------------------
    # Dimensionamento (usado para construir a rede neural compatível).
    # ------------------------------------------------------------------
    def num_saidas(self) -> int:
        return self.corpo.num_juntas()

    def num_entradas(self) -> int:
        # pés + ângulos de junta + up(3) + vel(3) + dir_alvo(3) + dist(1)
        return self._n_pes + self.corpo.num_juntas() + 3 + 3 + 3 + 1

    # ------------------------------------------------------------------
    # Sensores → vetor de entrada da rede.
    # ------------------------------------------------------------------
    def ler_sensores(self) -> List[float]:
        c = self.corpo
        entradas: List[float] = []

        # 1) toque dos pés no solo
        entradas.extend(1.0 if t else 0.0 for t in c.contatos_pe())

        # 2) ângulos atuais normalizados pelos limites da junta
        for i in range(c.num_juntas()):
            lo, hi = c.limites_junta(i)
            ang = c.angulo_junta(i)
            meio = 0.5 * (lo + hi)
            amp = max(1e-6, 0.5 * (hi - lo))
            entradas.append(clamp((ang - meio) / amp, -1.0, 1.0))

        # 3) orientação do giroscópio (eixo "up" do tronco)
        up = c.vetor_up_core()
        entradas.extend([up.x, up.y, up.z])

        # 4) velocidade linear do centro de massa
        v = c.velocidade_cm()
        entradas.extend([
            clamp(v.x * _ESCALA_VEL, -1.0, 1.0),
            clamp(v.y * _ESCALA_VEL, -1.0, 1.0),
            clamp(v.z * _ESCALA_VEL, -1.0, 1.0),
        ])

        # 5) vetor para o alvo (modo caça/caçador ou corrida com alvo)
        d = c.vetor_para_alvo()
        dist = d.length()
        dirn = d.normalized()
        entradas.extend([dirn.x, dirn.y, dirn.z])
        entradas.append(clamp(dist * _ESCALA_DIST, 0.0, 1.0))

        return entradas

    # ------------------------------------------------------------------
    # Aplicação do controle.
    # ------------------------------------------------------------------
    def aplicar_saidas(self, saidas: List[float]) -> None:
        c = self.corpo
        for m in self.musculos:
            alvo = saidas[m.junta_idx] if m.junta_idx < len(saidas) else 0.0
            ativacao = m.atualizar(alvo)
            c.definir_ativacao(m.junta_idx, ativacao)

    def passo_controle(self, tempo: float) -> None:
        """Executa um ciclo de controle (chamado a cada passo de física)."""
        assert self.controlador is not None, "Criatura sem cérebro."
        entradas = self.ler_sensores()
        saidas = self.controlador.avaliar(entradas, tempo)
        self.aplicar_saidas(saidas)

    # ------------------------------------------------------------------
    # Atualização de métricas pós-passo (para fitness de estabilidade/energia).
    # ------------------------------------------------------------------
    def acumular_metricas(self, energia_passo: float) -> None:
        self.energia_total += energia_passo
        up = self.corpo.vetor_up_core()
        # quanto o tronco se afasta da vertical (1=perfeito em pé).
        self.desvio_vertical_acc += (1.0 - clamp(up.y, -1.0, 1.0))
        v = self.corpo.velocidade_cm()
        self.oscilacao_lateral_acc += abs(v.z)
        self.passos += 1

    def reset_metricas(self) -> None:
        self.energia_total = 0.0
        self.desvio_vertical_acc = 0.0
        self.oscilacao_lateral_acc = 0.0
        self.passos = 0
        for m in self.musculos:
            m.reset()
