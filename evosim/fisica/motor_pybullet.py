"""Motor de física baseado em PyBullet (Bullet) — corpos rígidos articulados.

Este é o caminho de **física do estado da arte** (a mesma família usada pelos
ambientes de locomoção do MuJoCo/PyBullet Gym: Ant, HalfCheetah, Humanoid).
A criatura é um único *multibody articulado* (algoritmo de Featherstone):
elos rígidos ligados por juntas de revolução, atuados por **motores de junta
com torque limitado** (POSITION_CONTROL + maxForce). Isso modela um músculo de
forma fisicamente correta — diferente de reposicionar massa cinematicamente —
então não há injeção de energia nem movimento em espiral antinatural.

Requer:  pip install pybullet
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


# --- utilidades de quaternion via API do pybullet --------------------------
def _qmul(p, q1, q2):
    return p.multiplyTransforms([0, 0, 0], q1, [0, 0, 0], q2)[1]


def _qinv(p, q):
    return p.invertTransform([0, 0, 0], q)[1]


def _rot(p, q, v):
    return p.multiplyTransforms([0, 0, 0], q, list(v), [0, 0, 0, 1])[0]


def _quat_from_to(a: Vec3, b: Vec3):
    """Quaternion [x,y,z,w] que leva o vetor ``a`` ao vetor ``b``."""
    a = a.normalized()
    b = b.normalized()
    eixo = a.cross(b)
    ang = math.atan2(eixo.length(), a.dot(b))
    if eixo.length() < EPS:
        if a.dot(b) > 0:
            return [0.0, 0.0, 0.0, 1.0]
        eixo = Vec3(1, 0, 0) if abs(a.x) < 0.9 else Vec3(0, 1, 0)
    e = eixo.normalized()
    s = math.sin(ang / 2.0)
    return [e.x * s, e.y * s, e.z * s, math.cos(ang / 2.0)]


class _Junta:
    __slots__ = ("idx", "lo", "hi", "torque_max", "alvo_norm")

    def __init__(self, idx, lo, hi, torque):
        self.idx = idx
        self.lo = lo
        self.hi = hi
        self.torque_max = torque
        self.alvo_norm = 0.0


class CorpoPyBullet(CorpoCriatura):
    def __init__(self, motor: "MotorPyBullet", robot: int):
        self.motor = motor
        self.p = motor.p
        self.robot = robot
        self.juntas: List[_Junta] = []
        self.massas: List[float] = []        # massa de base + cada elo
        self.massa_total = 0.0
        self.pes: List[int] = []             # índices de elo dos pés
        self.proibidos: List[int] = []       # índices de elo proibidos (cabeça)
        self.todos_elos: List[int] = []      # -1 (base) + elos
        self.up_local = [0.0, 1.0, 0.0]      # "para cima" do corpo no repouso
        self._energia = 0.0
        self._alvo: Optional[Vec3] = None

    # --- juntas -------------------------------------------------------
    def num_juntas(self) -> int:
        return len(self.juntas)

    def angulo_junta(self, i: int) -> float:
        return self.p.getJointState(self.robot, self.juntas[i].idx)[0]

    def limites_junta(self, i: int) -> Tuple[float, float]:
        j = self.juntas[i]
        return (j.lo, j.hi)

    def definir_ativacao(self, i: int, valor_norm: float) -> None:
        self.juntas[i].alvo_norm = clamp(valor_norm, -1.0, 1.0)

    # --- estado -------------------------------------------------------
    def _pos_elo(self, idx: int) -> Vec3:
        if idx < 0:
            return Vec3(*self.p.getBasePositionAndOrientation(self.robot)[0])
        return Vec3(*self.p.getLinkState(self.robot, idx)[0])

    def centro_de_massa(self) -> Vec3:
        acc = Vec3(0, 0, 0)
        for idx, m in zip(self.todos_elos, self.massas):
            acc = acc + self._pos_elo(idx) * m
        return acc / self.massa_total if self.massa_total > EPS else Vec3()

    def pos_core(self) -> Vec3:
        return Vec3(*self.p.getBasePositionAndOrientation(self.robot)[0])

    def velocidade_cm(self) -> Vec3:
        return Vec3(*self.p.getBaseVelocity(self.robot)[0])

    def vetor_up_core(self) -> Vec3:
        _, orn = self.p.getBasePositionAndOrientation(self.robot)
        return Vec3(*_rot(self.p, orn, self.up_local))

    # --- contatos -----------------------------------------------------
    def _toca(self, idx: int) -> bool:
        return len(self.p.getContactPoints(bodyA=self.robot, linkIndexA=idx,
                                           bodyB=self.motor.solo)) > 0

    def contatos_pe(self) -> List[bool]:
        return [self._toca(i) for i in self.pes]

    def parte_proibida_no_solo(self) -> bool:
        return any(self._toca(i) for i in self.proibidos)

    def parte_nao_pe_no_solo(self) -> bool:
        pes = set(self.pes)
        return any(self._toca(i) for i in self.todos_elos if i not in pes)

    def energia_acumulada(self) -> float:
        return self._energia


class MotorPyBullet(MotorFisica):
    def __init__(self, ambiente: ConfigAmbiente, gui: bool = False):
        super().__init__(ambiente)
        import pybullet as p
        import pybullet_data
        self.p = p
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        g = ambiente.vetor_gravidade
        p.setGravity(g.x, g.y, g.z)
        p.setPhysicsEngineParameter(numSolverIterations=60,
                                    deterministicOverlappingPairs=1)
        self.solo = self._criar_solo()
        self.corpos: List[CorpoPyBullet] = []
        self.substeps = 4

    def _criar_solo(self) -> int:
        # Plano de chão com normal +Y (mundo Y-up, igual ao resto do sistema).
        # O plane.urdf padrão do PyBullet tem normal +Z (Z-up) e não serviria.
        p = self.p
        col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 1, 0])
        solo = p.createMultiBody(0, col,
                                 basePosition=[0, self.ambiente.altura_solo, 0])
        p.changeDynamics(solo, -1, lateralFriction=self.ambiente.friccao_solo,
                         restitution=self.ambiente.restituicao_solo)
        return solo

    def reset(self) -> None:
        self.p.resetSimulation()
        g = self.ambiente.vetor_gravidade
        self.p.setGravity(g.x, g.y, g.z)
        self.solo = self._criar_solo()
        self.corpos.clear()
        self.tempo = 0.0

    def fechar(self) -> None:
        try:
            self.p.disconnect(self.cid)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def construir_criatura(self, dna: CriaturaDNA, posicao_base: Vec3) -> CorpoPyBullet:
        dna.validar()
        p = self.p
        raiz = dna.raiz()

        # 1) pose de repouso: posição do nó proximal (junta) e direção de cada
        # segmento no mundo, percorrendo a árvore.
        prox_world: Dict[str, Vec3] = {}
        qworld: Dict[str, list] = {}
        prox_world[raiz.id] = Vec3(0, 0, 0)
        qworld[raiz.id] = _quat_from_to(Vec3(1, 0, 0), raiz.direcao_vec)

        def colocar(seg: SegmentoDNA):
            for filho in dna.filhos_de(seg.id):
                base = (prox_world[seg.id] if filho.conecta_em == "proximal"
                        else prox_world[seg.id] + seg.direcao_vec.normalized() * seg.comprimento)
                prox_world[filho.id] = base
                qworld[filho.id] = _quat_from_to(Vec3(1, 0, 0), filho.direcao_vec)
                colocar(filho)
        colocar(raiz)

        # 2) assenta no solo: desloca para o ponto mais baixo ficar acima do chão.
        extremos = []
        for sid in prox_world:
            seg = dna.por_id()[sid]
            extremos.append(prox_world[sid])
            extremos.append(prox_world[sid] + seg.direcao_vec.normalized() * seg.comprimento)
        miny = min(v.y for v in extremos)
        cx = sum(v.x for v in extremos) / len(extremos)
        cz = sum(v.z for v in extremos) / len(extremos)
        desloc = Vec3(posicao_base.x - cx,
                      self.ambiente.altura_solo + 0.12 - miny,
                      posicao_base.z - cz)

        # 3) ordem topológica dos elos (não-raiz) e índices.
        ordem: List[SegmentoDNA] = []
        def topo(seg):
            for f in dna.filhos_de(seg.id):
                ordem.append(f); topo(f)
        topo(raiz)
        idx_arr = {seg.id: i for i, seg in enumerate(ordem)}  # 0-based (=jointIndex)

        def capsula(seg: SegmentoDNA):
            r = max(0.03, seg.largura * 0.5)
            q_zx = _quat_from_to(Vec3(0, 0, 1), Vec3(1, 0, 0))  # cápsula Z -> X
            col = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=r, height=seg.comprimento,
                collisionFramePosition=[seg.comprimento * 0.5, 0, 0],
                collisionFrameOrientation=q_zx)
            return col

        # 4) monta os arrays do multibody.
        link_mass, link_col, link_vis = [], [], []
        link_pos, link_orn = [], []
        link_inertial_pos, link_inertial_orn = [], []
        link_parent, link_jtype, link_jaxis = [], [], []

        for seg in ordem:
            pai = dna.por_id()[seg.pai]
            # posição da junta no frame do PAI (eixo local +X do pai).
            if seg.conecta_em == "proximal":
                lp = [0.0, 0.0, 0.0]
            else:
                lp = [pai.comprimento, 0.0, 0.0]
            rel_orn = _qmul(p, _qinv(p, qworld[seg.pai]), qworld[seg.id])
            link_mass.append(seg.massa)
            link_col.append(capsula(seg))
            link_vis.append(-1)
            link_pos.append(lp)
            link_orn.append(rel_orn)
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parent.append(0 if seg.pai == raiz.id else idx_arr[seg.pai] + 1)
            link_jtype.append(p.JOINT_REVOLUTE)
            link_jaxis.append(list(seg.junta.eixo_vec))

        base_pos = (prox_world[raiz.id] + desloc).as_list()
        robot = p.createMultiBody(
            baseMass=raiz.massa,
            baseCollisionShapeIndex=capsula(raiz),
            basePosition=base_pos,
            baseOrientation=qworld[raiz.id],
            baseInertialFramePosition=[0, 0, 0],
            linkMasses=link_mass,
            linkCollisionShapeIndices=link_col,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=link_orn,
            linkInertialFramePositions=link_inertial_pos,
            linkInertialFrameOrientations=link_inertial_orn,
            linkParentIndices=link_parent,
            linkJointTypes=link_jtype,
            linkJointAxis=link_jaxis,
        )

        corpo = CorpoPyBullet(self, robot)
        p.changeDynamics(robot, -1, lateralFriction=self.ambiente.friccao_solo,
                         restitution=self.ambiente.restituicao_solo)
        # base = core; "para cima" do corpo no repouso (consistente entre presets).
        corpo.up_local = list(_rot(p, _qinv(p, qworld[raiz.id]), [0, 1, 0]))
        corpo.todos_elos = [-1] + list(range(len(ordem)))
        corpo.massas = [raiz.massa] + link_mass
        corpo.massa_total = raiz.massa + sum(link_mass)
        corpo.comprimentos = [raiz.comprimento] + [s.comprimento for s in ordem]
        if raiz.eh_pe:
            corpo.pes.append(-1)
        if raiz.proibido_solo:
            corpo.proibidos.append(-1)

        for seg in ordem:
            j = idx_arr[seg.id]
            p.changeDynamics(robot, j, lateralFriction=self.ambiente.friccao_solo,
                             restitution=self.ambiente.restituicao_solo,
                             jointLowerLimit=seg.junta.limite_min,
                             jointUpperLimit=seg.junta.limite_max,
                             linearDamping=self.ambiente.arrasto_fluido,
                             angularDamping=self.ambiente.arrasto_fluido)
            corpo.juntas.append(_Junta(j, seg.junta.limite_min,
                                       seg.junta.limite_max, seg.junta.torque_max))
            if seg.eh_pe:
                corpo.pes.append(j)
            if seg.proibido_solo:
                corpo.proibidos.append(j)

        self.corpos.append(corpo)
        return corpo

    # ------------------------------------------------------------------
    def passo(self, dt: float) -> None:
        p = self.p
        for corpo in self.corpos:
            for j in corpo.juntas:
                amp = 0.5 * (j.hi - j.lo)
                alvo = clamp(j.alvo_norm * amp, j.lo, j.hi)
                p.setJointMotorControl2(corpo.robot, j.idx, p.POSITION_CONTROL,
                                        targetPosition=alvo, force=j.torque_max)
        p.setTimeStep(dt / self.substeps)
        for _ in range(self.substeps):
            p.stepSimulation()
        for corpo in self.corpos:
            for j in corpo.juntas:
                tau = p.getJointState(corpo.robot, j.idx)[3]
                corpo._energia += abs(tau) * dt
        self.tempo += dt

    # ------------------------------------------------------------------
    def coletar_segmentos_render(self) -> List[Tuple[Vec3, Vec3, str]]:
        p = self.p
        linhas = []
        for corpo in self.corpos:
            for k, idx in enumerate(corpo.todos_elos):
                if idx < 0:
                    pos, orn = p.getBasePositionAndOrientation(corpo.robot)
                else:
                    st = p.getLinkState(corpo.robot, idx)
                    pos, orn = st[4], st[5]
                comp = corpo.comprimentos[k]
                a = Vec3(*pos)
                b = a + Vec3(*_rot(p, orn, [comp, 0, 0]))
                linhas.append((a, b, "seg"))
        return linhas
