"""Presets morfológicos de partida (reais e de fantasia).

Cada função devolve um ``CriaturaDNA`` pronto. São pontos de partida; a
evolução refina o controle (e, opcionalmente, proporções dentro dos limites).

Reais:     humanoide, quadrupede, hexapode, ave_bipede.
Fantasia:  centauro, voador_cauda, tripode.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

from .dna import CriaturaDNA, JuntaDNA, SegmentoDNA


def _j(eixo=(0.0, 0.0, 1.0), lo=-1.2, hi=1.2, torque=45.0, rig=0.4) -> JuntaDNA:
    return JuntaDNA(tipo="dobradica", eixo=list(eixo),
                    limite_min=lo, limite_max=hi, torque_max=torque, rigidez=rig)


def _perna(prefixo: str, pai: str, conecta: str, dir_coxa: Tuple[float, float, float],
           comp=(0.42, 0.42, 0.18), massa=(2.2, 1.6, 0.7),
           torque=55.0) -> List[SegmentoDNA]:
    """Membro de 3 segmentos: coxa -> canela -> pé."""
    coxa = SegmentoDNA(id=f"{prefixo}_coxa", papel="coxa", pai=pai, conecta_em=conecta,
                       direcao=list(dir_coxa), comprimento=comp[0], massa=massa[0],
                       junta=_j(lo=-1.3, hi=1.3, torque=torque))
    # Joelho: dobra só para TRÁS (anatômico). Repouso ~0 (perna reta), flexiona
    # no sentido negativo (a canela vai para trás), nunca para a frente.
    canela = SegmentoDNA(id=f"{prefixo}_canela", papel="canela", pai=coxa.id,
                         direcao=[0.0, -1.0, 0.0], comprimento=comp[1], massa=massa[1],
                         junta=_j(lo=-2.4, hi=0.05, torque=torque))
    pe = SegmentoDNA(id=f"{prefixo}_pe", papel="pe", pai=canela.id,
                     direcao=[0.8, -0.2, 0.0], comprimento=comp[2], massa=massa[2],
                     junta=_j(lo=-0.8, hi=0.8, torque=torque * 0.5))
    return [coxa, canela, pe]


# ---------------------------------------------------------------------------
def humanoide() -> CriaturaDNA:
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[0, 1, 0],
                            comprimento=0.55, largura=0.22, massa=8.0))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="core",
                            direcao=[0, 1, 0], comprimento=0.25, massa=1.2,
                            junta=_j(lo=-0.5, hi=0.5, torque=15)))
    # braços a partir dos ombros (extremidade distal do tronco).
    for lado, z in (("esq", 0.18), ("dir", -0.18)):
        segs.append(SegmentoDNA(id=f"braco_{lado}", papel="braco", pai="core",
                                direcao=[0.1, -1.0, z], comprimento=0.34, massa=1.4,
                                junta=_j(lo=-1.6, hi=1.6, torque=25)))
        segs.append(SegmentoDNA(id=f"antebraco_{lado}", papel="antebraco",
                                pai=f"braco_{lado}", direcao=[0, -1, 0],
                                comprimento=0.30, massa=1.0,
                                junta=_j(lo=-2.2, hi=0.0, torque=20)))
    # pernas a partir da pélvis (extremidade proximal do tronco).
    for lado, z in (("esq", 0.12), ("dir", -0.12)):
        segs += _perna(f"perna_{lado}", "core", "proximal", (0.0, -1.0, z))
    return CriaturaDNA("humanoide", "humanoide", segs, altura_inicial=1.1)


def quadrupede() -> CriaturaDNA:
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[1, 0, 0],
                            comprimento=0.8, largura=0.25, massa=10.0))
    segs.append(SegmentoDNA(id="pescoco", papel="pescoco", pai="core",
                            direcao=[1, 0.6, 0], comprimento=0.3, massa=1.5,
                            junta=_j(lo=-0.6, hi=0.6, torque=20)))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="pescoco",
                            direcao=[1, 0.2, 0], comprimento=0.22, massa=1.0,
                            junta=_j(lo=-0.4, hi=0.4, torque=12)))
    segs.append(SegmentoDNA(id="cauda", papel="cauda", pai="core", conecta_em="proximal",
                            direcao=[-1, 0.1, 0], comprimento=0.5, massa=1.2,
                            junta=_j(lo=-0.8, hi=0.8, torque=15)))
    # patas dianteiras (no ombro=distal) e traseiras (no quadril=proximal).
    for lado, z in (("esq", 0.18), ("dir", -0.18)):
        segs += _perna(f"diant_{lado}", "core", "distal", (0.1, -1.0, z))
        segs += _perna(f"tras_{lado}", "core", "proximal", (-0.1, -1.0, z))
    return CriaturaDNA("quadrupede", "quadrupede", segs, altura_inicial=0.9)


def hexapode() -> CriaturaDNA:
    """Inseto: tórax em 3 segmentos, um par de patas por segmento."""
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[1, 0, 0],
                            comprimento=0.35, largura=0.2, massa=4.0))
    segs.append(SegmentoDNA(id="tx2", papel="tronco", pai="core", direcao=[1, 0, 0],
                            comprimento=0.35, massa=3.0,
                            junta=_j(lo=-0.3, hi=0.3, torque=20)))
    segs.append(SegmentoDNA(id="tx3", papel="tronco", pai="tx2", direcao=[1, 0, 0],
                            comprimento=0.35, massa=3.0,
                            junta=_j(lo=-0.3, hi=0.3, torque=20)))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="core", conecta_em="proximal",
                            direcao=[-1, 0.2, 0], comprimento=0.2, massa=0.8,
                            junta=_j(lo=-0.3, hi=0.3, torque=8)))
    for seg_id in ("core", "tx2", "tx3"):
        for lado, z in (("esq", 1.0), ("dir", -1.0)):
            segs += _perna(f"{seg_id}_{lado}", seg_id, "distal", (0.0, -0.8, z),
                           comp=(0.3, 0.3, 0.12), massa=(0.8, 0.6, 0.3), torque=22)
    return CriaturaDNA("hexapode", "hexapode", segs, altura_inicial=0.55)


def ave_bipede() -> CriaturaDNA:
    """Ave/dinossauro bípede: pernas robustas, cauda contrabalançadora."""
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[0.4, 0.9, 0],
                            comprimento=0.5, largura=0.22, massa=7.0))
    segs.append(SegmentoDNA(id="pescoco", papel="pescoco", pai="core",
                            direcao=[0.6, 0.8, 0], comprimento=0.3, massa=1.2,
                            junta=_j(lo=-0.7, hi=0.7, torque=15)))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="pescoco",
                            direcao=[1, 0.2, 0], comprimento=0.22, massa=0.9,
                            junta=_j(lo=-0.4, hi=0.4, torque=10)))
    segs.append(SegmentoDNA(id="cauda", papel="cauda", pai="core", conecta_em="proximal",
                            direcao=[-1, -0.1, 0], comprimento=0.7, massa=2.0,
                            junta=_j(lo=-0.6, hi=0.6, torque=22)))
    for lado, z in (("esq", 0.13), ("dir", -0.13)):
        segs += _perna(f"perna_{lado}", "core", "proximal", (0.0, -1.0, z),
                       comp=(0.5, 0.5, 0.22), massa=(3.0, 2.0, 0.9), torque=70)
    return CriaturaDNA("ave_bipede", "ave_bipede", segs, altura_inicial=1.0)


def centauro() -> CriaturaDNA:
    """Fantasia: tronco quadrúpede + torso humanoide com braços."""
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[1, 0, 0],
                            comprimento=0.8, largura=0.25, massa=11.0))
    # torso ereto na frente (no ombro do cavalo).
    segs.append(SegmentoDNA(id="torso", papel="tronco", pai="core", direcao=[0.2, 1, 0],
                            comprimento=0.5, massa=5.0,
                            junta=_j(lo=-0.5, hi=0.5, torque=30)))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="torso", direcao=[0, 1, 0],
                            comprimento=0.22, massa=1.0,
                            junta=_j(lo=-0.4, hi=0.4, torque=12)))
    for lado, z in (("esq", 0.18), ("dir", -0.18)):
        segs.append(SegmentoDNA(id=f"braco_{lado}", papel="braco", pai="torso",
                                direcao=[0.2, -0.6, z], comprimento=0.32, massa=1.2,
                                junta=_j(lo=-1.6, hi=1.6, torque=22)))
        segs.append(SegmentoDNA(id=f"antebraco_{lado}", papel="antebraco",
                                pai=f"braco_{lado}", direcao=[0.3, -0.8, 0],
                                comprimento=0.28, massa=0.9,
                                junta=_j(lo=-2.0, hi=0.0, torque=18)))
    segs.append(SegmentoDNA(id="cauda", papel="cauda", pai="core", conecta_em="proximal",
                            direcao=[-1, 0.1, 0], comprimento=0.5, massa=1.2,
                            junta=_j(lo=-0.8, hi=0.8, torque=15)))
    for lado, z in (("esq", 0.18), ("dir", -0.18)):
        segs += _perna(f"diant_{lado}", "core", "distal", (0.1, -1.0, z))
        segs += _perna(f"tras_{lado}", "core", "proximal", (-0.1, -1.0, z))
    return CriaturaDNA("centauro", "centauro", segs, altura_inicial=1.2)


def voador_cauda() -> CriaturaDNA:
    """Fantasia: criatura alada com longa cauda propulsora segmentada."""
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[1, 0, 0],
                            comprimento=0.5, largura=0.3, massa=5.0))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="core",
                            direcao=[1, 0.2, 0], comprimento=0.22, massa=0.8,
                            junta=_j(lo=-0.4, hi=0.4, torque=10)))
    for lado, z in (("esq", 1.0), ("dir", -1.0)):
        segs.append(SegmentoDNA(id=f"asa_{lado}", papel="asa", pai="core",
                                conecta_em="proximal", direcao=[0.0, 0.1, z],
                                comprimento=0.7, largura=0.4, massa=1.0,
                                junta=_j(eixo=(1, 0, 0), lo=-1.0, hi=1.0, torque=40)))
    # cauda propulsora: cadeia de 4 segmentos.
    pai = "core"
    for k in range(4):
        sid = f"cauda{k}"
        segs.append(SegmentoDNA(id=sid, papel="cauda", pai=pai,
                                conecta_em="proximal" if k == 0 else "distal",
                                direcao=[-1, 0.0, 0], comprimento=0.35, massa=0.7,
                                junta=_j(eixo=(0, 1, 0), lo=-0.9, hi=0.9, torque=25)))
        pai = sid
    return CriaturaDNA("voador_cauda", "voador_cauda", segs, altura_inicial=0.6)


def tripode() -> CriaturaDNA:
    """Fantasia: três pernas radiais a 120° em torno de um hub central."""
    segs: List[SegmentoDNA] = []
    segs.append(SegmentoDNA(id="core", papel="core", direcao=[0, 1, 0],
                            comprimento=0.4, largura=0.25, massa=6.0))
    segs.append(SegmentoDNA(id="cabeca", papel="cabeca", pai="core",
                            direcao=[0, 1, 0], comprimento=0.2, massa=0.8,
                            junta=_j(lo=-0.4, hi=0.4, torque=10)))
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        dx, dz = math.cos(ang), math.sin(ang)
        segs += _perna(f"perna{k}", "core", "proximal", (dx * 0.6, -1.0, dz * 0.6),
                       comp=(0.45, 0.45, 0.2), massa=(2.0, 1.5, 0.7), torque=55)
    return CriaturaDNA("tripode", "tripode", segs, altura_inicial=0.95)


PRESETS: Dict[str, Callable[[], CriaturaDNA]] = {
    "humanoide": humanoide,
    "quadrupede": quadrupede,
    "hexapode": hexapode,
    "ave_bipede": ave_bipede,
    "centauro": centauro,
    "voador_cauda": voador_cauda,
    "tripode": tripode,
}


def criar_preset(nome: str) -> CriaturaDNA:
    if nome not in PRESETS:
        raise ValueError(f"Preset desconhecido: {nome}. Opções: {list(PRESETS)}")
    return PRESETS[nome]()


def listar_presets() -> List[str]:
    return list(PRESETS)
