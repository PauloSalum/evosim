"""Gerenciador de evolução para a UI web.

Roda o treino em uma thread de fundo, permite parar de forma cooperativa,
expõe status/histórico e os frames do campeão de cada geração (para o
visualizador 3D ao vivo), e suporta:
* iniciar do zero a partir de um preset;
* CONTINUAR a partir de um save (warm-start);
* editar o ambiente (gravidade etc.) a cada novo treino ou no playback.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from ..config import ConfigAmbiente, ConfigSimulacao
from ..criaturas.presets import criar_preset
from ..evolucao import criar_algoritmo
from ..persistencia.serializacao import Save, carregar, salvar
from ..simulacao.executor import Executor
from . import sim_api


class _Parar(Exception):
    """Sinaliza parada cooperativa do laço de evolução."""


def _ambiente_de(params: dict) -> ConfigAmbiente:
    g = params.get("gravidade", [0.0, -9.81, 0.0])
    return ConfigAmbiente(
        gravidade=list(g),
        arrasto_fluido=float(params.get("arrasto_fluido", 0.02)),
        friccao_solo=float(params.get("friccao_solo", 0.9)),
        restituicao_solo=float(params.get("restituicao_solo", 0.0)),
    )


class GerenciadorEvolucao:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._parar = False
        self.rodando = False
        self.historico: list = []
        self.geracao = -1
        self.preset = ""
        self.frames: dict = {"frames": [], "dt": 0.05, "preset": ""}
        self.executor: Optional[Executor] = None
        self.ambiente_atual: dict = ConfigAmbiente().__dict__.copy()
        self.erro = ""
        self.concluido = False
        self.autosave_path = ""
        self.autosave_geracao = -1
        self.ilimitado = False
        self.motor_nome = "auto"

    # ------------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            return {
                "rodando": self.rodando,
                "concluido": self.concluido,
                "geracao": self.geracao,
                "preset": self.preset,
                "historico": list(self.historico),
                "erro": self.erro,
                "ambiente": self.ambiente_atual,
                "autosave": self.autosave_path,
                "autosave_geracao": self.autosave_geracao,
                "ilimitado": self.ilimitado,
                "tem_melhor": self.executor is not None
                and self.executor.melhor_atual() is not None,
            }

    def frames_status(self) -> dict:
        with self._lock:
            return dict(self.frames)

    # ------------------------------------------------------------------
    def iniciar(self, params: dict) -> dict:
        with self._lock:
            if self.rodando:
                return {"ok": False, "erro": "Já existe um treino em andamento."}
            self._parar = False
            self.rodando = True
            self.concluido = False
            self.erro = ""
            self.historico = []
            self.geracao = -1
        self._thread = threading.Thread(target=self._rodar, args=(params,), daemon=True)
        self._thread.start()
        return {"ok": True}

    def parar(self) -> dict:
        with self._lock:
            self._parar = True
        return {"ok": True}

    def salvar(self, caminho: str) -> dict:
        with self._lock:
            ex = self.executor
        if ex is None or ex.melhor_atual() is None:
            return {"ok": False, "erro": "Nada para salvar ainda."}
        salvar(ex.para_save(), caminho)
        return {"ok": True, "caminho": caminho}

    # ------------------------------------------------------------------
    def _rodar(self, params: dict) -> None:
        try:
            ambiente = _ambiente_de(params)
            sim = ConfigSimulacao(
                seed=int(params.get("seed", 1234)),
                duracao_segundos=float(params.get("duracao", 8.0)),
            )
            modo = params.get("modo", "novo")
            geracoes = int(params.get("geracoes", 0))  # 0 => ilimitado
            pop = int(params.get("pop", 24))
            workers = int(params.get("workers", 0))
            algoritmo = params.get("algoritmo", "es")
            fitness = params.get("fitness", "velocidade")
            controlador = params.get("controlador", "cpg")
            autosave_cada = max(1, int(params.get("autosave_cada", 1)))
            motor_nome = params.get("motor", "auto")
            genoma_inicial = None

            if modo == "continuar":
                save = carregar(params["save_path"])
                dna = save.dna_obj()
                genoma_inicial = save.melhor_genoma()
                preset = save.preset
                fitness = params.get("fitness") or save.fitness_nome
            else:
                preset = params["preset"]
                dna = criar_preset(preset)

            # caminho do checkpoint (auto-save a cada geração).
            autosave = params.get("autosave") or f"runs/{preset}_auto.json"
            os.makedirs(os.path.dirname(autosave) or ".", exist_ok=True)

            with self._lock:
                self.preset = preset
                self.ambiente_atual = ambiente.__dict__.copy()
                self.autosave_path = autosave
                self.autosave_cada = autosave_cada
                self.autosave_geracao = -1
                self.ilimitado = geracoes <= 0
                self.motor_nome = motor_nome

            factory = criar_algoritmo(algoritmo, tamanho_pop=pop, seed=sim.seed)
            executor = Executor(
                dna, ambiente, sim, factory, fitness_nome=fitness,
                tipo_controlador=controlador, n_workers=workers,
                genoma_inicial=genoma_inicial, motor=motor_nome,
            )
            with self._lock:
                self.executor = executor

            executor.evoluir(geracoes, callback=self._on_geracao,
                             monitor=self._on_monitor)
            with self._lock:
                self.concluido = True
        except _Parar:
            with self._lock:
                self.concluido = True
        except Exception as e:  # registra o erro para o frontend
            with self._lock:
                self.erro = f"{type(e).__name__}: {e}"
        finally:
            with self._lock:
                self.rodando = False

    def _on_geracao(self, ger: int, stats: dict) -> None:
        with self._lock:
            self.geracao = ger
            self.historico.append(stats)
            cada = getattr(self, "autosave_cada", 1)
            caminho = self.autosave_path
            ex = self.executor
        # checkpoint: salva o melhor-até-agora a cada N gerações, fora do lock.
        if caminho and ex is not None and ger % cada == 0:
            try:
                salvar(ex.para_save(), caminho)
                with self._lock:
                    self.autosave_geracao = ger
            except Exception:
                pass

    def _on_monitor(self, executor: Executor, ger: int, stats: dict) -> None:
        # gera os frames do campeão desta geração (clip curto p/ visualização).
        melhor = executor.melhor_atual()
        if melhor is not None:
            clip = ConfigSimulacao(duracao_segundos=min(executor.sim.duracao_segundos, 6.0))
            frames = sim_api.simular_frames(
                executor.dna, melhor.instanciar_controlador(),
                executor.ambiente, clip, cada=3, segundos=clip.duracao_segundos,
                motor_nome=self.motor_nome,
            )
            with self._lock:
                self.frames = {
                    "frames": frames, "dt": clip.dt * 3,
                    "preset": self.preset, "geracao": ger,
                    "fitness": stats.get("melhor", 0.0),
                }
        with self._lock:
            parar = self._parar
        if parar:
            raise _Parar()

    # ------------------------------------------------------------------
    def playback(self, override_ambiente: dict, segundos: float = 8.0) -> dict:
        """Reproduz o melhor atual com um ambiente sobreposto (gravidade etc.)."""
        with self._lock:
            ex = self.executor
        if ex is None or ex.melhor_atual() is None:
            return {"frames": [], "dt": 0.05, "erro": "Sem indivíduo treinado."}
        ambiente = sim_api.ambiente_com_override(ex.ambiente.__dict__, override_ambiente)
        clip = ConfigSimulacao(duracao_segundos=segundos)
        frames = sim_api.simular_frames(
            ex.dna, ex.melhor_atual().instanciar_controlador(),
            ambiente, clip, cada=2, segundos=segundos, motor_nome=self.motor_nome,
        )
        return {"frames": frames, "dt": clip.dt * 2, "preset": self.preset}
