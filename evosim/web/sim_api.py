"""Ponte simulação → frames JSON para o frontend web.

Roda a criatura no ``MotorInterno`` (o mesmo motor do treino) e devolve os
segmentos quadro a quadro como listas de floats, prontos para o visualizador
3D do navegador. Permite sobrepor parâmetros do ambiente (ex.: gravidade) em
tempo real sem retreinar.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.criatura import Criatura
from ..criaturas.dna import CriaturaDNA
from ..fisica.motor_interno import MotorInterno
from ..mathutils import Vec3
from ..neural.controlador import ControladorNeural
from ..persistencia.serializacao import Save

# Cada frame: lista de segmentos; cada segmento = [ax, ay, az, bx, by, bz].
Frame = List[List[float]]


def _terminou(corpo, sim: ConfigSimulacao, cm: Vec3) -> bool:
    """Mesmos critérios de parada precoce do treino — para o visualizador
    mostrar exatamente o que o treino recompensa (e parar na cambalhota/queda)."""
    if cm.y < sim.altura_critica:
        return True
    if corpo.vetor_up_core().y < sim.up_minimo_capotar:
        return True
    if sim.apenas_pes_no_solo and corpo.parte_nao_pe_no_solo():
        return True
    if sim.proibe_contato_cabeca and corpo.parte_proibida_no_solo():
        return True
    return False


def simular_frames(
    dna: CriaturaDNA,
    controlador: ControladorNeural,
    ambiente: ConfigAmbiente,
    sim: ConfigSimulacao,
    cada: int = 2,
    segundos: Optional[float] = None,
    parar_cedo: bool = True,
) -> List[Frame]:
    motor = MotorInterno(ambiente)
    motor.substeps = sim.substeps
    cri = Criatura(dna, motor.construir_criatura(dna, Vec3(0, 0, 0)))
    cri.controlador = controlador
    passos = sim.passos_totais if segundos is None else int(segundos / sim.dt)
    frames: List[Frame] = []
    for p in range(passos):
        cri.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        if p % cada == 0:
            frames.append([
                [a.x, a.y, a.z, b.x, b.y, b.z]
                for a, b, _ in motor.coletar_segmentos_render()
            ])
        # encerra a animação no mesmo ponto em que o treino encerraria.
        if parar_cedo and _terminou(cri.corpo, sim, cri.corpo.centro_de_massa()):
            break
    return frames


def ambiente_com_override(base: dict, override: Optional[dict]) -> ConfigAmbiente:
    dados = dict(base) if base else {}
    if override:
        dados.update({k: v for k, v in override.items() if v is not None})
    # remove chaves desconhecidas para não quebrar o dataclass.
    validos = ConfigAmbiente().__dict__.keys()
    dados = {k: v for k, v in dados.items() if k in validos}
    return ConfigAmbiente(**dados)


def _snapshot_multi(motor) -> List[List[float]]:
    """Segmentos de TODAS as criaturas do motor, cada um marcado com o índice
    da criatura (7º valor) para colorir grupos diferentes no 3D."""
    out: List[List[float]] = []
    for ci, corpo in enumerate(motor.corpos):
        nos = set(corpo.nos)
        for (a, b, _r) in motor.bones:
            if a in nos and b in nos:
                pa, pb = motor.pos[a], motor.pos[b]
                out.append([pa.x, pa.y, pa.z, pb.x, pb.y, pb.z, ci])
    return out


def rodar_corrida_web(
    saves: List[Save], override_ambiente: Optional[dict] = None,
    segundos: float = 10.0, cada: int = 2,
) -> dict:
    """Corrida entre vários saves: anima todos e devolve o ranking."""
    from ..modos.corrida import Competidor, rodar_corrida
    ambiente = ambiente_com_override(ConfigAmbiente().__dict__, override_ambiente)
    sim = ConfigSimulacao(duracao_segundos=segundos)
    comps = [Competidor.de_save(s) for s in saves]
    frames: List[List[List[float]]] = []

    def hook(motor, passo):
        if passo % cada == 0:
            frames.append(_snapshot_multi(motor))

    ranking = rodar_corrida(comps, ambiente=ambiente, sim=sim, on_step=hook)
    return {
        "frames": frames, "dt": sim.dt * cada,
        "grupos": [c.nome for c in comps],
        "ranking": [{"nome": n, "dist": round(d, 2)} for n, d in ranking],
    }


def rodar_caca_web(
    save_cacador: Save, save_presa: Save,
    override_ambiente: Optional[dict] = None, segundos: float = 10.0, cada: int = 2,
) -> dict:
    """Episódio de caça entre dois saves: anima e devolve o desfecho."""
    from ..modos.caca_cacador import rodar_episodio
    ambiente = ambiente_com_override(ConfigAmbiente().__dict__, override_ambiente)
    sim = ConfigSimulacao(duracao_segundos=segundos)
    dna_c = save_cacador.dna_obj()
    dna_p = save_presa.dna_obj()
    ctrl_c = save_cacador.melhor_genoma().instanciar_controlador()
    ctrl_p = save_presa.melhor_genoma().instanciar_controlador()
    frames: List[List[List[float]]] = []

    def hook(motor, passo):
        if passo % cada == 0:
            frames.append(_snapshot_multi(motor))

    res = rodar_episodio(dna_c, ctrl_c, dna_p, ctrl_p, ambiente, sim, on_step=hook)
    return {
        "frames": frames, "dt": sim.dt * cada,
        "grupos": [f"caçador ({save_cacador.preset})", f"presa ({save_presa.preset})"],
        "resultado": {
            "capturou": res.capturou,
            "tempo": round(res.tempo, 2),
            "distancia": round(res.distancia_final, 2),
        },
    }


def frames_de_save(
    save: Save,
    override_ambiente: Optional[dict] = None,
    segundos: float = 8.0,
    cada: int = 2,
) -> dict:
    """Simula o melhor indivíduo de um save e devolve frames + metadados."""
    ambiente = ambiente_com_override(save.ambiente, override_ambiente)
    sim = ConfigSimulacao(duracao_segundos=segundos)
    dna = save.dna_obj()
    ctrl = save.melhor_genoma().instanciar_controlador()
    frames = simular_frames(dna, ctrl, ambiente, sim, cada=cada, segundos=segundos)
    return {
        "preset": save.preset,
        "dt": sim.dt * cada,
        "frames": frames,
    }
