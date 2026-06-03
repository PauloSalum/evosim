"""Adaptador para PyBullet — engine de corpos rígidos robusta (opcional).

Implementa a mesma interface ``MotorFisica`` / ``CorpoCriatura`` do motor
interno, porém com física de corpos rígidos completa (colisões, atrito,
restituição reais). Use-o quando quiser fidelidade máxima:

    pip install pybullet

Modelo de coordenadas maximais: cada segmento é um corpo rígido (cápsula)
conectado ao pai por uma restrição ponto-a-ponto; os músculos aplicam torque
limitado no eixo da junta (análogo ao motor interno). A simulação roda em
**passo fixo** (``setTimeStep`` + ``setPhysicsEngineParameter`` determinístico).

Este módulo não tem dependência em tempo de import: o ``import pybullet`` só
ocorre ao instanciar ``MotorPyBullet``, então o resto do sistema funciona sem
ele instalado.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..config import ConfigAmbiente
from ..criaturas.dna import CriaturaDNA, SegmentoDNA
from ..mathutils import EPS, Vec3, clamp
from .motor import CorpoCriatura, MotorFisica


def disponivel() -> bool:
    try:
        import pybullet  # noqa: F401
        return True
    except Exception:
        return False


def _quat_de_para(p, origem: Vec3, destino: Vec3):
    """Quaternion que rotaciona ``origem`` para ``destino`` (ambos unitários)."""
    a = origem.normalized()
    b = destino.normalized()
    eixo = a.cross(b)
    ang = math.atan2(eixo.length(), a.dot(b))
    if eixo.length() < EPS:
        eixo = Vec3(1, 0, 0) if abs(a.x) < 0.9 else Vec3(0, 1, 0)
    e = eixo.normalized()
    return p.getQuaternionFromAxisAngle([e.x, e.y, e.z], ang) \
        if hasattr(p, "getQuaternionFromAxisAngle") else _quat_axis_angle(e, ang)


def _quat_axis_angle(e: Vec3, ang: float):
    s = math.sin(ang / 2.0)
    return [e.x * s, e.y * s, e.z * s, math.cos(ang / 2.0)]


class _JuntaPB:
    __slots__ = ("corpo_filho", "corpo_pai", "eixo", "lo", "hi",
                 "torque_max", "rigidez", "alvo_norm", "rest")

    def __init__(self):
        self.corpo_filho = 0
        self.corpo_pai = 0
        self.eixo = Vec3(0, 0, 1)
        self.lo = -math.pi
        self.hi = math.pi
        self.torque_max = 40.0
        self.rigidez = 0.4
        self.alvo_norm = 0.0
        self.rest = 0.0


class CorpoPyBullet(CorpoCriatura):
    def __init__(self, motor: "MotorPyBullet"):
        self.motor = motor
        self.p = motor.p
        self.bodies: Dict[str, int] = {}
        self.massa_total = 0.0
        self.massas: Dict[str, float] = {}
        self.juntas: List[_JuntaPB] = []
        self.id_core = -1
        self.pes: List[int] = []        # bodyIds dos segmentos-pé
        self.proibidos: List[int] = []  # bodyIds proibidos de tocar o solo
        self._energia = 0.0
        self._alvo: Optional[Vec3] = None

    # --- juntas -------------------------------------------------------
    def num_juntas(self) -> int:
        return len(self.juntas)

    def _orient_forward(self, body: int) -> Vec3:
        _, orn = self.p.getBasePositionAndOrientation(body)
        m = self.p.getMatrixFromQuaternion(orn)
        return Vec3(m[2], m[5], m[8])  # eixo Z local em coordenadas do mundo

    def angulo_junta(self, i: int) -> float:
        j = self.juntas[i]
        fp = self._orient_forward(j.corpo_pai)
        fc = self._orient_forward(j.corpo_filho)
        n = j.eixo.normalized()
        return math.atan2(fp.cross(fc).dot(n), fp.dot(fc)) - j.rest

    def limites_junta(self, i: int) -> Tuple[float, float]:
        j = self.juntas[i]
        return (j.lo, j.hi)

    def definir_ativacao(self, i: int, valor_norm: float) -> None:
        self.juntas[i].alvo_norm = clamp(valor_norm, -1.0, 1.0)

    # --- estado -------------------------------------------------------
    def centro_de_massa(self) -> Vec3:
        acc = Vec3(0, 0, 0)
        for sid, body in self.bodies.items():
            pos, _ = self.p.getBasePositionAndOrientation(body)
            acc = acc + Vec3(*pos) * self.massas[sid]
        return acc / self.massa_total if self.massa_total > EPS else Vec3()

    def pos_core(self) -> Vec3:
        pos, _ = self.p.getBasePositionAndOrientation(self.id_core)
        return Vec3(*pos)

    def velocidade_cm(self) -> Vec3:
        acc = Vec3(0, 0, 0)
        for sid, body in self.bodies.items():
            v, _ = self.p.getBaseVelocity(body)
            acc = acc + Vec3(*v) * self.massas[sid]
        return acc / self.massa_total if self.massa_total > EPS else Vec3()

    def vetor_up_core(self) -> Vec3:
        _, orn = self.p.getBasePositionAndOrientation(self.id_core)
        m = self.p.getMatrixFromQuaternion(orn)
        return Vec3(m[1], m[4], m[7])  # eixo Y local no mundo

    def contatos_pe(self) -> List[bool]:
        out = []
        for body in self.pes:
            pts = self.p.getContactPoints(bodyA=body, bodyB=self.motor.solo)
            out.append(len(pts) > 0)
        return out

    def parte_proibida_no_solo(self) -> bool:
        for body in self.proibidos:
            if self.p.getContactPoints(bodyA=body, bodyB=self.motor.solo):
                return True
        return False

    def parte_nao_pe_no_solo(self) -> bool:
        pes = set(self.pes)
        for body in self.bodies.values():
            if body in pes:
                continue
            if self.p.getContactPoints(bodyA=body, bodyB=self.motor.solo):
                return True
        return False

    def energia_acumulada(self) -> float:
        return self._energia


class MotorPyBullet(MotorFisica):
    def __init__(self, ambiente: ConfigAmbiente, gui: bool = False):
        super().__init__(ambiente)
        import pybullet as p
        import pybullet_data
        self.p = p
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        g = ambiente.vetor_gravidade
        p.setGravity(g.x, g.y, g.z)
        # Passo fixo e determinístico (não usar setRealTimeSimulation).
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0, numSolverIterations=50,
            deterministicOverlappingPairs=1,
        )
        self.solo = p.loadURDF("plane.urdf")
        p.changeDynamics(self.solo, -1,
                         lateralFriction=ambiente.friccao_solo,
                         restitution=ambiente.restituicao_solo)
        self.corpos: List[CorpoPyBullet] = []
        self.substeps = 4

    def reset(self) -> None:
        self.p.resetSimulation()
        g = self.ambiente.vetor_gravidade
        self.p.setGravity(g.x, g.y, g.z)
        self.solo = self.p.loadURDF("plane.urdf")
        self.corpos.clear()
        self.tempo = 0.0

    # ------------------------------------------------------------------
    def construir_criatura(self, dna: CriaturaDNA, posicao_base: Vec3) -> CorpoPyBullet:
        dna.validar()
        p = self.p
        corpo = CorpoPyBullet(self)

        # posições dos nós (igual ao motor interno) para situar cada cápsula.
        por_id = dna.por_id()
        no_pos: Dict[str, Tuple[Vec3, Vec3]] = {}  # seg.id -> (prox, distal)
        raiz = dna.raiz()

        def coloca(seg: SegmentoDNA, prox: Vec3):
            distal = prox + seg.direcao_vec.normalized() * seg.comprimento
            no_pos[seg.id] = (prox, distal)
            for filho in dna.filhos_de(seg.id):
                attach = prox if filho.conecta_em == "proximal" else distal
                coloca(filho, attach)

        coloca(raiz, Vec3(0, 0, 0))

        # reposiciona para assentar no solo + base x/z.
        todos = [v for pr in no_pos.values() for v in pr]
        miny = min(v.y for v in todos)
        cx = sum(v.x for v in todos) / len(todos)
        cz = sum(v.z for v in todos) / len(todos)
        desloc = Vec3(posicao_base.x - cx,
                      self.ambiente.altura_solo + 0.08 - miny,
                      posicao_base.z - cz)

        for seg in dna.segmentos:
            prox, distal = no_pos[seg.id]
            prox = prox + desloc
            distal = distal + desloc
            centro = (prox + distal) * 0.5
            r = max(0.02, seg.largura * 0.5)
            col = p.createCollisionShape(p.GEOM_CAPSULE, radius=r,
                                         height=seg.comprimento)
            orn = _quat_de_para(p, Vec3(0, 0, 1), distal - prox)
            body = p.createMultiBody(
                baseMass=seg.massa, baseCollisionShapeIndex=col,
                basePosition=[centro.x, centro.y, centro.z], baseOrientation=orn,
            )
            p.changeDynamics(body, -1, lateralFriction=self.ambiente.friccao_solo,
                             restitution=self.ambiente.restituicao_solo,
                             linearDamping=self.ambiente.arrasto_fluido)
            corpo.bodies[seg.id] = body
            corpo.massas[seg.id] = seg.massa
            corpo.massa_total += seg.massa
            if seg.papel == "core":
                corpo.id_core = body
            if seg.eh_pe:
                corpo.pes.append(body)
            if seg.proibido_solo:
                corpo.proibidos.append(body)

        # restrições de junta (ponto-a-ponto na posição do nó compartilhado).
        for seg in dna.segmentos:
            if seg.pai is None:
                continue
            prox, _ = no_pos[seg.id]
            pivot = prox + desloc
            pai_body = corpo.bodies[seg.pai]
            filho_body = corpo.bodies[seg.id]
            self._ponto_a_ponto(corpo, pai_body, filho_body, pivot)
            j = _JuntaPB()
            j.corpo_pai = pai_body
            j.corpo_filho = filho_body
            j.eixo = seg.junta.eixo_vec
            j.lo, j.hi = seg.junta.limite_min, seg.junta.limite_max
            j.torque_max = seg.junta.torque_max
            j.rigidez = seg.junta.rigidez
            corpo.juntas.append(j)

        # rest angles
        for i, j in enumerate(corpo.juntas):
            fp = corpo._orient_forward(j.corpo_pai)
            fc = corpo._orient_forward(j.corpo_filho)
            n = j.eixo.normalized()
            j.rest = math.atan2(fp.cross(fc).dot(n), fp.dot(fc))

        self.corpos.append(corpo)
        return corpo

    def _ponto_a_ponto(self, corpo, pai_body, filho_body, pivot_world: Vec3):
        p = self.p
        ppos, porn = p.getBasePositionAndOrientation(pai_body)
        cpos, corn = p.getBasePositionAndOrientation(filho_body)
        inv_p = p.invertTransform(ppos, porn)
        inv_c = p.invertTransform(cpos, corn)
        piv_p, _ = p.multiplyTransforms(inv_p[0], inv_p[1],
                                        [pivot_world.x, pivot_world.y, pivot_world.z],
                                        [0, 0, 0, 1])
        piv_c, _ = p.multiplyTransforms(inv_c[0], inv_c[1],
                                        [pivot_world.x, pivot_world.y, pivot_world.z],
                                        [0, 0, 0, 1])
        p.createConstraint(pai_body, -1, filho_body, -1, p.JOINT_POINT2POINT,
                           [0, 0, 0], piv_p, piv_c)

    # ------------------------------------------------------------------
    def passo(self, dt: float) -> None:
        p = self.p
        for corpo in self.corpos:
            for j in corpo.juntas:
                self._aplicar_musculo(corpo, j)
        # garante consistência com o dt fixo configurado.
        p.setTimeStep(dt / self.substeps)
        for _ in range(self.substeps):
            p.stepSimulation()
        self.tempo += dt

    def _aplicar_musculo(self, corpo: CorpoPyBullet, j: _JuntaPB) -> None:
        p = self.p
        atual = corpo.angulo_junta(corpo.juntas.index(j))
        amp = 0.5 * (j.hi - j.lo)
        alvo = clamp(j.alvo_norm * amp, j.lo, j.hi)
        erro = alvo - atual
        torque = clamp(erro * j.rigidez * j.torque_max, -j.torque_max, j.torque_max)
        n = j.eixo.normalized()
        vetor = [n.x * torque, n.y * torque, n.z * torque]
        p.applyExternalTorque(j.corpo_filho, -1, vetor, p.WORLD_FRAME)
        p.applyExternalTorque(j.corpo_pai, -1,
                              [-vetor[0], -vetor[1], -vetor[2]], p.WORLD_FRAME)
        corpo._energia += abs(torque) * 0.01

    def coletar_segmentos_render(self) -> List[Tuple[Vec3, Vec3, str]]:
        linhas = []
        for corpo in self.corpos:
            for sid, body in corpo.bodies.items():
                pos, orn = self.p.getBasePositionAndOrientation(body)
                m = self.p.getMatrixFromQuaternion(orn)
                fwd = Vec3(m[2], m[5], m[8])
                c = Vec3(*pos)
                linhas.append((c - fwd * 0.2, c + fwd * 0.2, sid))
        return linhas
