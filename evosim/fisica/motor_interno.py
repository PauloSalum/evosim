"""Motor de física interno, determinístico, em Python puro.

Implementa um solver baseado em posição (estilo PBD/XPBD) sobre um sistema de
partículas. Cada criatura é um esqueleto de "varetas": segmentos rígidos são
restrições de distância entre nós; os músculos são atuadores angulares que
giram a subárvore de um membro em torno da junta, com torque limitado.

Vantagens para este projeto:
* **Sem dependências nativas** — roda em qualquer máquina com Python.
* **Determinístico** — passo de tempo fixo + ordem de operações estável.
* **Estável** — projeção de restrições não explode como integração de forças.

É uma aproximação (juntas esféricas são tratadas como dobradiças em torno de um
eixo fixo). Para fidelidade de corpos rígidos completos, use ``MotorPyBullet``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..config import ConfigAmbiente
from ..criaturas.dna import CriaturaDNA, SegmentoDNA
from ..mathutils import EPS, Vec3, clamp
from .motor import CorpoCriatura, MotorFisica

MAX_VEL = 14.0
_CONTATO_EPS = 0.03
# Velocidade angular máxima de um músculo (rad/s). Limita o quão rápido um
# membro pode girar — sem isso os membros "chicoteiam" e a reação do solo
# catapulta a criatura (parece que não há gravidade).
MAX_OMEGA = 6.0
# Teto para a velocidade vertical (m/s) de um nó em contato com o solo. Impede
# que um pé batendo no chão lance o corpo para o alto (efeito catapulta).
MAX_VY_CONTATO = 3.0
# Teto para a velocidade de SUBIDA do centro de massa da criatura (m/s). Garante
# que nenhuma criatura "voe": no máximo um pulo curto (~v²/2g de altura).
MAX_VY_CM = 2.5


class _Junta:
    """Estado interno de uma junta/músculo do motor interno."""

    __slots__ = (
        "idx_b", "idx_prox_pai", "idx_child_distal", "subarvore",
        "eixo", "rest", "lo", "hi", "torque_max", "rigidez", "alvo_norm",
    )

    def __init__(self) -> None:
        self.idx_b: int = 0
        self.idx_prox_pai: int = 0
        self.idx_child_distal: int = 0
        self.subarvore: List[int] = []
        self.eixo: Vec3 = Vec3(0, 0, 1)
        self.rest: float = 0.0
        self.lo: float = -math.pi
        self.hi: float = math.pi
        self.torque_max: float = 40.0
        self.rigidez: float = 0.35
        self.alvo_norm: float = 0.0


class CorpoInterno(CorpoCriatura):
    def __init__(self, motor: "MotorInterno") -> None:
        self.motor = motor
        self.juntas: List[_Junta] = []
        self.nos: List[int] = []                  # todos os índices de nós
        self.nos_pe: List[List[int]] = []         # nós de cada segmento-pé
        self.nos_proibidos: List[int] = []        # nós de segmentos proibidos
        self.idx_core_prox: int = 0
        self.idx_core_distal: int = 0
        self.idx_core_up: int = 0
        self._energia: float = 0.0
        self._alvo: Optional[Vec3] = None

    # --- juntas -------------------------------------------------------
    def num_juntas(self) -> int:
        return len(self.juntas)

    def _angulo_atual(self, j: _Junta) -> float:
        p = self.motor.pos
        b = p[j.idx_b]
        pai_dir = (b - p[j.idx_prox_pai]).normalized()
        child_dir = (p[j.idx_child_distal] - b).normalized()
        n = j.eixo.normalized()
        cruz = pai_dir.cross(child_dir)
        return math.atan2(cruz.dot(n), pai_dir.dot(child_dir))

    def angulo_junta(self, i: int) -> float:
        return self._angulo_atual(self.juntas[i])

    def limites_junta(self, i: int) -> Tuple[float, float]:
        j = self.juntas[i]
        return (j.lo, j.hi)

    def definir_ativacao(self, i: int, valor_norm: float) -> None:
        self.juntas[i].alvo_norm = clamp(valor_norm, -1.0, 1.0)

    # --- estado -------------------------------------------------------
    def centro_de_massa(self) -> Vec3:
        return self.motor._cm_de(self.nos)

    def pos_core(self) -> Vec3:
        p = self.motor.pos
        return (p[self.idx_core_prox] + p[self.idx_core_distal]) * 0.5

    def velocidade_cm(self) -> Vec3:
        return self.motor._vcm_de(self.nos)

    def vetor_up_core(self) -> Vec3:
        p = self.motor.pos
        meio = (p[self.idx_core_prox] + p[self.idx_core_distal]) * 0.5
        return (p[self.idx_core_up] - meio).normalized()

    def contatos_pe(self) -> List[bool]:
        p = self.motor.pos
        chao = self.motor.ambiente.altura_solo + _CONTATO_EPS
        return [any(p[n].y <= chao for n in grupo) for grupo in self.nos_pe]

    def parte_proibida_no_solo(self) -> bool:
        p = self.motor.pos
        chao = self.motor.ambiente.altura_solo + _CONTATO_EPS
        return any(p[n].y <= chao for n in self.nos_proibidos)

    def energia_acumulada(self) -> float:
        return self._energia


class MotorInterno(MotorFisica):
    def __init__(self, ambiente: ConfigAmbiente) -> None:
        super().__init__(ambiente)
        self.pos: List[Vec3] = []
        self.vel: List[Vec3] = []
        self.inv_mass: List[float] = []
        self.massa: List[float] = []
        self.bones: List[Tuple[int, int, float]] = []
        self.corpos: List[CorpoInterno] = []
        self._iteracoes = 6

    def reset(self) -> None:
        self.pos.clear(); self.vel.clear(); self.inv_mass.clear()
        self.massa.clear(); self.bones.clear(); self.corpos.clear()
        self.tempo = 0.0

    # ------------------------------------------------------------------
    # Construção da criatura a partir do DNA.
    # ------------------------------------------------------------------
    def _novo_no(self, p: Vec3) -> int:
        idx = len(self.pos)
        self.pos.append(p)
        self.vel.append(Vec3(0, 0, 0))
        self.massa.append(0.0)
        self.inv_mass.append(0.0)
        return idx

    def construir_criatura(self, dna: CriaturaDNA, posicao_base: Vec3) -> CorpoInterno:
        dna.validar()
        corpo = CorpoInterno(self)
        por_id = dna.por_id()
        raiz = dna.raiz()

        # 1) Posiciona nós percorrendo a árvore (pose de repouso no mundo).
        no_distal: Dict[str, int] = {}
        no_prox: Dict[str, int] = {}

        base = self._novo_no(Vec3(0, 0, 0))  # core proximal na origem temporária
        corpo.idx_core_prox = base
        no_prox[raiz.id] = base
        corpo.nos.append(base)

        def construir_seg(seg: SegmentoDNA, prox_idx: int) -> None:
            pos_prox = self.pos[prox_idx]
            distal = self._novo_no(pos_prox + seg.direcao_vec.normalized() * seg.comprimento)
            no_prox[seg.id] = prox_idx
            no_distal[seg.id] = distal
            corpo.nos.append(distal)
            # massa distribuída meio-a-meio entre os dois nós do segmento.
            self.massa[prox_idx] += seg.massa * 0.5
            self.massa[distal] += seg.massa * 0.5
            self.bones.append((prox_idx, distal, seg.comprimento))
            if seg.eh_pe:
                corpo.nos_pe.append([distal, prox_idx])
            if seg.proibido_solo:
                corpo.nos_proibidos.append(distal)
            for filho in dna.filhos_de(seg.id):
                # filho pode se ligar à extremidade proximal ou distal deste seg.
                attach = prox_idx if filho.conecta_em == "proximal" else distal
                construir_seg(filho, attach)

        construir_seg(raiz, base)
        corpo.idx_core_distal = no_distal[raiz.id]

        # Nó de referência "up" rígido no core (para giroscópio/equilíbrio).
        meio_core = (self.pos[corpo.idx_core_prox] + self.pos[corpo.idx_core_distal]) * 0.5
        up_idx = self._novo_no(meio_core + Vec3(0, 0.3, 0))
        corpo.idx_core_up = up_idx
        corpo.nos.append(up_idx)
        self.massa[up_idx] += 0.05
        self.bones.append((corpo.idx_core_prox, up_idx,
                           self.pos[corpo.idx_core_prox].distance(self.pos[up_idx])))
        self.bones.append((corpo.idx_core_distal, up_idx,
                           self.pos[corpo.idx_core_distal].distance(self.pos[up_idx])))

        # 2) Constrói juntas/músculos (todo segmento não-raiz tem um).
        for seg in dna.segmentos:
            if seg.pai is None:
                continue
            j = _Junta()
            b = no_prox[seg.id]                   # vértice = nó de fixação no pai
            # referência = a OUTRA extremidade do pai (define a direção do pai).
            pp, pd = no_prox[seg.pai], no_distal[seg.pai]
            ref = pp if b == pd else pd
            j.idx_b = b
            j.idx_prox_pai = ref
            j.idx_child_distal = no_distal[seg.id]
            j.subarvore = self._coletar_subarvore(dna, seg, no_distal)
            jd = seg.junta
            j.eixo = jd.eixo_vec
            j.lo, j.hi = jd.limite_min, jd.limite_max
            j.torque_max = jd.torque_max
            j.rigidez = jd.rigidez
            corpo.juntas.append(j)

        # 3) Finaliza massas/inv_mass e reposiciona no mundo.
        for n in corpo.nos:
            m = self.massa[n]
            self.inv_mass[n] = (1.0 / m) if m > EPS else 0.0

        self._assentar_no_solo(corpo, posicao_base)

        # rest angle de cada junta na pose final.
        for j in corpo.juntas:
            j.rest = corpo._angulo_atual(j)

        self.corpos.append(corpo)
        return corpo

    def _coletar_subarvore(self, dna, seg, no_distal) -> List[int]:
        """Índices de nós que devem girar rigidamente com este membro."""
        ids: List[int] = []
        pilha = [seg]
        while pilha:
            s = pilha.pop()
            ids.append(no_distal[s.id])
            pilha.extend(dna.filhos_de(s.id))
        return ids

    def _assentar_no_solo(self, corpo: CorpoInterno, base: Vec3) -> None:
        ys = [self.pos[n].y for n in corpo.nos]
        xs = [self.pos[n].x for n in corpo.nos]
        zs = [self.pos[n].z for n in corpo.nos]
        cx = sum(xs) / len(xs)
        cz = sum(zs) / len(zs)
        desloc = Vec3(
            base.x - cx,
            self.ambiente.altura_solo + 0.05 - min(ys),
            base.z - cz,
        )
        for n in corpo.nos:
            self.pos[n] = self.pos[n] + desloc

    # ------------------------------------------------------------------
    # Massa / centro de massa.
    # ------------------------------------------------------------------
    def _cm_de(self, nos: List[int]) -> Vec3:
        total = 0.0
        acc = Vec3(0, 0, 0)
        for n in nos:
            m = self.massa[n]
            acc = acc + self.pos[n] * m
            total += m
        return acc / total if total > EPS else Vec3(0, 0, 0)

    def _vcm_de(self, nos: List[int]) -> Vec3:
        total = 0.0
        acc = Vec3(0, 0, 0)
        for n in nos:
            m = self.massa[n]
            acc = acc + self.vel[n] * m
            total += m
        return acc / total if total > EPS else Vec3(0, 0, 0)

    # ------------------------------------------------------------------
    # Passo de simulação (fixed timestep com subpassos).
    # ------------------------------------------------------------------
    def passo(self, dt: float) -> None:
        sub = max(1, self._subpassos())
        sdt = dt / sub
        g = self.ambiente.vetor_gravidade
        arrasto = self.ambiente.arrasto_fluido
        for _ in range(sub):
            self._substep(sdt, g, arrasto)
        self.tempo += dt

    def _subpassos(self) -> int:
        # Definido pelo avaliador a partir de ConfigSimulacao; default razoável.
        return getattr(self, "substeps", 4)

    def _substep(self, sdt: float, g: Vec3, arrasto: float) -> None:
        """Um subpasso PBD: prediz, projeta restrições, recupera velocidade."""
        pos, vel, im = self.pos, self.vel, self.inv_mass
        n = len(pos)
        damp = max(0.0, 1.0 - arrasto * sdt)
        prev = list(pos)  # cópia das posições antes da predição.

        # 1) predição (integração explícita de velocidade -> posição).
        for i in range(n):
            if im[i] == 0.0:
                continue
            v = (vel[i] + g * sdt) * damp
            if v.length_sq() > MAX_VEL * MAX_VEL:
                v = v.normalized() * MAX_VEL
            vel[i] = v
            pos[i] = pos[i] + v * sdt

        # 2) restrições de posição: músculos, ossos e solo (não-penetração).
        self._aplicar_musculos(sdt)
        for _ in range(self._iteracoes):
            self._resolver_bones()
        chao = self.ambiente.altura_solo
        for i in range(n):
            if im[i] == 0.0:
                continue
            if pos[i].y < chao:
                pos[i] = Vec3(pos[i].x, chao, pos[i].z)

        # 3) recupera velocidade a partir do deslocamento corrigido.
        inv = 1.0 / sdt
        for i in range(n):
            if im[i] == 0.0:
                continue
            vel[i] = (pos[i] - prev[i]) * inv

        # 4) atrito de Coulomb + restituição nos nós em contato.
        self._atrito_solo()

        # 5) garantia anti-voo: limita a velocidade de subida do CoM de cada
        # criatura (forças internas não levantam o corpo; só um pulo curto).
        self._limitar_subida_cm()

    def _atrito_solo(self) -> None:
        chao = self.ambiente.altura_solo
        fr = self.ambiente.friccao_solo
        rest = self.ambiente.restituicao_solo
        pos, vel, im = self.pos, self.vel, self.inv_mass
        for i in range(len(pos)):
            if im[i] == 0.0 or pos[i].y > chao + _CONTATO_EPS:
                continue
            v = vel[i]
            vy = -rest * v.y if v.y < 0.0 else v.y
            # impede catapulta: nó em contato não dispara para cima.
            if vy > MAX_VY_CONTATO:
                vy = MAX_VY_CONTATO
            vt = Vec3(v.x, 0.0, v.z)
            vtl = vt.length()
            if vtl > EPS:
                reduz = min(vtl, fr * max(abs(v.y), 0.5))
                vt = vt * ((vtl - reduz) / vtl)
            vel[i] = Vec3(vt.x, vy, vt.z)

    def _limitar_subida_cm(self) -> None:
        for corpo in self.corpos:
            vcm = self._vcm_de(corpo.nos)
            if vcm.y > MAX_VY_CM:
                excesso = vcm.y - MAX_VY_CM
                for n in corpo.nos:
                    v = self.vel[n]
                    self.vel[n] = Vec3(v.x, v.y - excesso, v.z)

    def _resolver_bones(self) -> None:
        pos, im = self.pos, self.inv_mass
        for (a, b, rest) in self.bones:
            wa, wb = im[a], im[b]
            w = wa + wb
            if w == 0.0:
                continue
            delta = pos[b] - pos[a]
            d = delta.length()
            if d < EPS:
                continue
            corr = (d - rest) / d
            pos[a] = pos[a] + delta * (corr * wa / w)
            pos[b] = pos[b] - delta * (corr * wb / w)

    def _aplicar_musculos(self, sdt: float) -> None:
        pos = self.pos
        for corpo in self.corpos:
            # CoM antes da atuação: músculos são forças INTERNAS e não podem
            # deslocar o centro de massa (só gravidade e o contato com o solo,
            # que são externos, podem). Medimos antes/depois e cancelamos o
            # deslocamento espúrio — sem isso, girar a sub-árvore de um membro
            # injeta energia e a criatura "voa".
            cm_antes = self._cm_de(corpo.nos)
            for j in corpo.juntas:
                atual = corpo._angulo_atual(j)
                amp = 0.5 * (j.hi - j.lo)
                alvo = clamp(j.rest + j.alvo_norm * amp, j.lo, j.hi)
                erro = alvo - atual
                # Velocidade angular limitada (rad/s): músculos mais fortes giram
                # um pouco mais rápido, mas SEMPRE com um teto biológico. Isto
                # impede o chicoteamento que catapultava a criatura.
                omega_max = MAX_OMEGA * (0.5 + 0.5 * j.torque_max / 40.0)
                passo_max = omega_max * sdt
                delta = clamp(erro * j.rigidez, -passo_max, passo_max)
                if abs(delta) < 1e-6:
                    continue
                b = pos[j.idx_b]
                eixo = j.eixo
                for n in j.subarvore:
                    rel = pos[n] - b
                    pos[n] = b + rel.rotate_around(eixo, delta)
                corpo._energia += abs(delta) * j.torque_max
            cm_depois = self._cm_de(corpo.nos)
            desloc = cm_depois - cm_antes
            if desloc.length_sq() > 1e-18:
                for n in corpo.nos:
                    pos[n] = pos[n] - desloc

    # ------------------------------------------------------------------
    # Render snapshot.
    # ------------------------------------------------------------------
    def coletar_segmentos_render(self) -> List[Tuple[Vec3, Vec3, str]]:
        linhas: List[Tuple[Vec3, Vec3, str]] = []
        for corpo in self.corpos:
            for (a, b, _r) in self.bones:
                if a in corpo.nos and b in corpo.nos:
                    linhas.append((self.pos[a], self.pos[b], "seg"))
        return linhas
