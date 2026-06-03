"""Configuração do ambiente e da simulação.

Estes parâmetros descrevem o *mundo* e o orçamento de simulação. Eles são
**fixos** durante a evolução: o algoritmo evolutivo nunca pode alterá-los
(ver ``evosim/evolucao``). Isso garante que o fitness reflita a criatura e
não uma exploração das regras do ambiente.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from .mathutils import Vec3, vec

try:  # YAML é opcional; JSON é sempre suportado.
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


@dataclass
class ConfigAmbiente:
    """Parâmetros físicos do mundo (imutáveis para a evolução)."""

    gravidade: List[float] = field(default_factory=lambda: [0.0, -9.81, 0.0])
    # Coeficiente de arrasto linear do fluido (ar/água). 0 = vácuo.
    arrasto_fluido: float = 0.02
    # Densidade do fluido (usada por motores que modelam empuxo/asas).
    densidade_fluido: float = 1.225
    # Atrito de Coulomb do solo.
    friccao_solo: float = 0.9
    # Restituição (quão "elástico" é o solo). 0 = sem pulo.
    restituicao_solo: float = 0.0
    altura_solo: float = 0.0

    @property
    def vetor_gravidade(self) -> Vec3:
        return vec(self.gravidade)


@dataclass
class ConfigSimulacao:
    """Orçamento temporal e critérios de parada precoce."""

    # Passo de tempo FIXO da física (independente do FPS de renderização).
    dt: float = 1.0 / 120.0
    # Subpassos do solver de restrições por passo de física (estabilidade).
    substeps: int = 4
    # Duração máxima da avaliação de um indivíduo, em segundos simulados.
    duracao_segundos: float = 12.0
    # --- parada precoce (early termination) ---
    altura_critica: float = 0.35        # CoM abaixo disso => caiu.
    janela_aquecimento: float = 2.0     # tempo antes de checar deslocamento.
    deslocamento_minimo: float = 0.25   # se não andar isso após aquecimento, aborta.
    proibe_contato_cabeca: bool = True  # cabeça tocar o chão aborta.
    # Só os pés podem tocar o solo: qualquer outra parte encostando aborta
    # ("se jogar machuca"). Evita rastejar/usar o corpo como apoio.
    apenas_pes_no_solo: bool = True
    # Penaliza cambalhotas/capotamento: o "para cima" do tronco precisa ficar
    # acima deste valor (1 = perfeitamente em pé, 0 = deitado de lado). Com 0.2
    # a criatura pode inclinar até ~78°, mas dar uma cambalhota a mata.
    up_minimo_capotar: float = 0.2
    seed: int = 1234

    @property
    def passos_totais(self) -> int:
        return int(round(self.duracao_segundos / self.dt))


# ---------------------------------------------------------------------------
# Carregamento / serialização de configs.
# ---------------------------------------------------------------------------
def _load_text(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML não instalado; use JSON.")
            return yaml.safe_load(fh) or {}
        return json.load(fh)


def carregar_ambiente(path: str) -> ConfigAmbiente:
    return ConfigAmbiente(**_load_text(path))


def carregar_simulacao(path: str) -> ConfigSimulacao:
    return ConfigSimulacao(**_load_text(path))


def salvar_config(cfg: Any, path: str) -> None:
    data = asdict(cfg)
    with open(path, "w", encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")) and _HAS_YAML:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, fh, indent=2, ensure_ascii=False)
