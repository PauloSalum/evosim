"""Persistência da evolução (save/load em JSON/YAML).

Um arquivo de save é autocontido: guarda a morfologia (preset/DNA), os
parâmetros do ambiente usados no treino, a configuração de fitness, a geração
atual, o histórico de métricas e os melhores genomas (pesos da rede). Isso
permite recarregar criaturas treinadas independentemente e fazê-las competir
no Sandbox / Caça-Caçador.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..criaturas.dna import CriaturaDNA
from ..evolucao.genoma import Genoma

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

VERSAO_SAVE = 1


@dataclass
class Save:
    preset: str
    dna: dict
    ambiente: dict
    fitness_nome: str = "velocidade"
    geracao: int = 0
    historico: List[dict] = field(default_factory=list)
    melhores: List[dict] = field(default_factory=list)
    versao: int = VERSAO_SAVE

    # --- conveniências ------------------------------------------------
    def dna_obj(self) -> CriaturaDNA:
        return CriaturaDNA.from_dict(self.dna)

    def melhor_genoma(self) -> Genoma:
        if not self.melhores:
            raise ValueError("Save sem genomas.")
        return Genoma.from_dict(self.melhores[0])

    def to_dict(self) -> dict:
        return {
            "versao": self.versao,
            "preset": self.preset,
            "fitness_nome": self.fitness_nome,
            "geracao": self.geracao,
            "ambiente": self.ambiente,
            "dna": self.dna,
            "historico": self.historico,
            "melhores": self.melhores,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Save":
        return cls(
            preset=data["preset"],
            dna=data["dna"],
            ambiente=data.get("ambiente", {}),
            fitness_nome=data.get("fitness_nome", "velocidade"),
            geracao=data.get("geracao", 0),
            historico=data.get("historico", []),
            melhores=data.get("melhores", []),
            versao=data.get("versao", VERSAO_SAVE),
        )


def salvar(save: Save, path: str) -> None:
    data = save.to_dict()
    with open(path, "w", encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")) and _HAS_YAML:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, fh, indent=2, ensure_ascii=False)


def carregar(path: str) -> Save:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML não instalado; use JSON.")
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    return Save.from_dict(data)
