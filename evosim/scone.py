"""Integração com o SCONE (sconegym) — músculos Hill reais + modelos OpenSim.

Este módulo conecta a nossa evolução (CMA-ES via ``EstrategiaEvolutiva`` + uma
política ``RedeNeuralMLP``) aos ambientes musculoesqueléticos do **sconegym**,
do Thomas Geijtenbeek — os mesmos modelos (H0918, H1622, H2190) com músculos
Hill-type roteados que produzem a marcha natural dos vídeos do SCONE/Hyfydy.

A política mapeia a observação do modelo (comprimentos/forças musculares,
orientação da cabeça, contato dos pés, DOFs...) em **excitações musculares**
em [0, 1]. O *reward* do sconegym já é o objetivo no estilo SCONE (velocidade-
alvo + suavidade + esforço + penalidades), então a fitness é a soma do reward.

Requisitos (instalar na SUA máquina — não vêm via pip):
  1. SCONE:        https://scone.software  (com Hyfydy ou OpenSim)
  2. sconegym:     git clone https://github.com/tgeijten/sconegym
                   cd sconegym && pip install -r requirements.txt && pip install -e .

Depois:
  python -m evosim.cli scone --env sconewalk_h0918-v1 --geracoes 200 --pop 16 \
      --saida runs/scone_h0918.json
O melhor episódio é gravado em formato SCONE (.sto) para você assistir no
SCONE Studio (com os músculos coloridos).
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .evolucao.algoritmos import EstrategiaEvolutiva
from .evolucao.genoma import Genoma
from .mathutils import mean
from .neural.mlp import RedeNeuralMLP

# IDs de ambiente registrados pelo sconegym (caminhar e correr).
AMBIENTES = [
    "sconewalk_h0918-v1", "sconewalk_h1622-v1", "sconewalk_h2190-v1",
    "sconewalk_h0918_osim-v1",
    "sconerun_h0918-v1", "sconerun_h1622-v1", "sconerun_h2190-v1",
]


def disponivel() -> bool:
    try:
        import sconegym  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Compatibilidade gym 0.13 (usado pelo sconegym) e gymnasium.
# ---------------------------------------------------------------------------
def _make(env_id: str):
    import sconegym  # noqa: F401  (importar registra os ambientes)
    try:
        import gym
        return gym.make(env_id)
    except Exception:
        import gymnasium as gym
        return gym.make(env_id)


def _reset(env, seed: Optional[int] = None):
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:  # gymnasium
        obs, r, term, trunc, info = out
        return obs, r, bool(term or trunc), info
    return out  # gym clássico: obs, r, done, info


# ---------------------------------------------------------------------------
# Avaliação de uma política num episódio do sconegym.
# ---------------------------------------------------------------------------
def avaliar_politica(env, controlador, max_passos: int = 2000,
                     seed: Optional[int] = None) -> float:
    import numpy as np
    obs = _reset(env, seed)
    total = 0.0
    for _ in range(max_passos):
        # MLP devolve [-1, 1]; excitação muscular é [0, 1].
        saida = controlador.avaliar([float(x) for x in obs], 0.0)
        action = np.clip([(o + 1.0) * 0.5 for o in saida], 0.0, 1.0)
        obs, r, done, _info = _step(env, action)
        total += float(r)
        if done:
            break
    return total


# ---------------------------------------------------------------------------
# Treino: CMA-ES otimizando os pesos da MLP sobre o reward do SCONE.
# ---------------------------------------------------------------------------
def _carregar_spec(origem) -> dict:
    """Aceita um dict (já carregado) ou um caminho para o JSON da política."""
    if isinstance(origem, dict):
        return origem
    import json
    with open(origem, "r", encoding="utf-8") as fh:
        return json.load(fh)


def treinar(
    env_id: str = "sconewalk_h0918-v1",
    geracoes: int = 200,
    tamanho_pop: int = 16,
    ocultas: Optional[List[int]] = None,
    seed: int = 1234,
    callback: Optional[Callable[[int, dict], None]] = None,
    gravar_episodio: bool = True,
    continuar_de=None,
    max_passos: int = 2000,
) -> dict:
    if not disponivel():
        raise RuntimeError(
            "sconegym/SCONE não instalados. Veja docs/SCONE.md e o topo de "
            "evosim/scone.py (https://scone.software e github.com/tgeijten/sconegym)."
        )
    from .neural.controlador import from_dict
    env = _make(env_id)
    n_obs = int(env.observation_space.shape[0])
    n_act = int(env.action_space.shape[0])

    # protótipo: do zero, ou continuando de uma política salva (warm-start).
    semente = None
    if continuar_de is not None:
        spec = _carregar_spec(continuar_de)
        proto = from_dict(spec["controlador"])
        if proto.num_entradas != n_obs or proto.num_saidas != n_act:
            raise RuntimeError(
                f"Política salva ({proto.num_entradas}->{proto.num_saidas}) não "
                f"casa com o ambiente {env_id} ({n_obs}->{n_act}).")
        semente = proto.get_pesos()
    else:
        proto = RedeNeuralMLP(n_obs, n_act, ocultas=ocultas or [64, 64], seed=seed)

    algo = EstrategiaEvolutiva(proto, tamanho_pop=tamanho_pop, seed=seed)
    if semente is not None:
        algo.semear(semente)

    historico: List[dict] = []
    pop = algo.inicializar()
    for ger in range(geracoes):
        for g in pop:
            g.fitness = avaliar_politica(env, g.instanciar_controlador(),
                                         max_passos=max_passos, seed=seed + ger)
        fits = [g.fitness for g in pop]
        stats = {"geracao": ger, "melhor": max(fits), "media": mean(fits)}
        historico.append(stats)
        if callback:
            callback(ger, stats)
        pop = algo.proxima_geracao(pop)

    algo._registrar_melhor(pop)
    melhor = algo.melhor

    # grava o melhor episódio no formato SCONE (.sto/.par) para o SCONE Studio.
    if gravar_episodio and melhor is not None:
        try:
            env.store_next_episode()
            avaliar_politica(env, melhor.instanciar_controlador(),
                             max_passos=max_passos, seed=seed)
            env.write_now()
        except Exception as e:
            print(f"   (aviso: não consegui gravar o episódio SCONE: {e})")

    return {
        "env": env_id,
        "geracao": geracoes,
        "historico": historico,
        "controlador": melhor.controlador_spec if melhor else proto.to_dict(),
        "fitness": melhor.fitness if melhor else float("-inf"),
    }


def reproduzir(origem, env_id: Optional[str] = None, seed: int = 1234,
               max_passos: int = 2000) -> float:
    """Carrega uma política salva e grava UM episódio SCONE para assistir no
    SCONE Studio. Devolve o reward total."""
    if not disponivel():
        raise RuntimeError("sconegym/SCONE não instalados (ver docs/SCONE.md).")
    spec = _carregar_spec(origem)
    env = _make(env_id or spec.get("env", "sconewalk_h0918-v1"))
    ctrl = carregar_politica(spec)
    try:
        env.store_next_episode()
        total = avaliar_politica(env, ctrl, max_passos=max_passos, seed=seed)
        env.write_now()
        return total
    finally:
        pass


def carregar_politica(spec: dict) -> RedeNeuralMLP:
    from .neural.controlador import from_dict
    return from_dict(spec["controlador"])
