"""DNA morfológico das criaturas (serializável em JSON/YAML).

Uma criatura é uma **árvore** de segmentos rígidos. O segmento raiz é o
tronco/pélvis (``core``); cada outro segmento se liga a um pai por uma junta
(esférica ou dobradiça) acionada por um músculo. Este DNA descreve a
morfologia e seus *limites* — nunca o comportamento, que vive na rede neural.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from ..mathutils import Vec3, vec


# Papéis reconhecidos pelos sensores / critérios de parada precoce.
PAPEL_CORE = "core"
PAPEL_CABECA = "cabeca"
PAPEL_PE = "pe"
PAPEIS_VALIDOS = {
    PAPEL_CORE, PAPEL_CABECA, PAPEL_PE,
    "coxa", "canela", "braco", "antebraco", "asa", "cauda", "pescoco", "tronco",
}


@dataclass
class JuntaDNA:
    """Especificação da junta que conecta um segmento ao seu pai."""

    tipo: str = "dobradica"            # 'dobradica' | 'esferica'
    eixo: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    limite_min: float = -math.pi / 2   # radianos
    limite_max: float = math.pi / 2
    torque_max: float = 40.0           # N·m (teto do músculo)
    rigidez: float = 0.35              # fração da correção por passo (0..1)

    @property
    def eixo_vec(self) -> Vec3:
        return vec(self.eixo)


@dataclass
class SegmentoDNA:
    """Um osso/segmento rígido da criatura."""

    id: str
    papel: str = "tronco"
    pai: Optional[str] = None
    # Direção de repouso do segmento a partir do nó do pai (no espaço do mundo
    # na pose inicial). Define a postura de "T-pose" da criatura.
    direcao: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    comprimento: float = 0.4
    largura: float = 0.1
    massa: float = 1.0
    # Em qual extremidade do pai este segmento se conecta: a distal (padrão) ou
    # a proximal. Permite, p.ex., pernas saindo do quadril enquanto a cabeça
    # sai do topo do mesmo tronco.
    conecta_em: str = "distal"          # 'distal' | 'proximal'
    junta: Optional[JuntaDNA] = None   # None apenas para o core (raiz).
    # Genes morfológicos opcionais que a evolução PODE ajustar dentro de
    # [escala_min, escala_max] se ``ConfigEvolucao.evoluir_morfologia`` estiver on.
    escala_min: float = 1.0
    escala_max: float = 1.0

    @property
    def direcao_vec(self) -> Vec3:
        return vec(self.direcao)

    @property
    def proibido_solo(self) -> bool:
        return self.papel == PAPEL_CABECA

    @property
    def eh_pe(self) -> bool:
        return self.papel == PAPEL_PE


@dataclass
class CriaturaDNA:
    """Morfologia completa: árvore de segmentos a partir de um ``core``."""

    nome: str
    preset: str
    segmentos: List[SegmentoDNA] = field(default_factory=list)
    altura_inicial: float = 1.0  # altura do core ao spawnar.

    # --- consultas ----------------------------------------------------
    def por_id(self) -> Dict[str, SegmentoDNA]:
        return {s.id: s for s in self.segmentos}

    def raiz(self) -> SegmentoDNA:
        for s in self.segmentos:
            if s.pai is None:
                return s
        raise ValueError("DNA inválido: nenhum segmento raiz (core).")

    def filhos_de(self, sid: str) -> List[SegmentoDNA]:
        return [s for s in self.segmentos if s.pai == sid]

    def num_musculos(self) -> int:
        return sum(1 for s in self.segmentos if s.junta is not None)

    def validar(self) -> None:
        ids = {s.id for s in self.segmentos}
        if len(ids) != len(self.segmentos):
            raise ValueError("IDs de segmento duplicados.")
        raizes = [s for s in self.segmentos if s.pai is None]
        if len(raizes) != 1:
            raise ValueError("DNA precisa de exatamente 1 segmento raiz.")
        for s in self.segmentos:
            if s.pai is not None and s.pai not in ids:
                raise ValueError(f"Segmento {s.id} referencia pai inexistente {s.pai}.")
            if s.papel not in PAPEIS_VALIDOS:
                raise ValueError(f"Papel desconhecido: {s.papel}")
            if s.pai is not None and s.junta is None:
                raise ValueError(f"Segmento não-raiz {s.id} precisa de junta.")

    # --- serialização -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CriaturaDNA":
        segs = []
        for sd in data.get("segmentos", []):
            jd = sd.get("junta")
            junta = JuntaDNA(**jd) if jd else None
            seg_kwargs = {k: v for k, v in sd.items() if k != "junta"}
            segs.append(SegmentoDNA(junta=junta, **seg_kwargs))
        return cls(
            nome=data["nome"],
            preset=data["preset"],
            segmentos=segs,
            altura_inicial=data.get("altura_inicial", 1.0),
        )

    def copia(self) -> "CriaturaDNA":
        return CriaturaDNA.from_dict(self.to_dict())
