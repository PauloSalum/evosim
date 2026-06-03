"""Atuador muscular com suavização (interpolação de ativação).

Controladores neurais tendem a produzir saídas que mudam bruscamente a cada
passo, o que gera vibrações frenéticas e irreais. Um músculo real tem inércia
e elasticidade: não muda de tensão instantaneamente. Modelamos isso filtrando
a saída do controlador com um LERP de primeira ordem (filtro passa-baixa),
parametrizado por ``suavizacao`` (0 = músculo travado, 1 = resposta imediata).
"""
from __future__ import annotations

from ..mathutils import clamp, lerp


class Musculo:
    def __init__(self, junta_idx: int, suavizacao: float = 0.18) -> None:
        self.junta_idx = junta_idx
        self.suavizacao = clamp(suavizacao, 0.0, 1.0)
        self.ativacao: float = 0.0  # estado interno suavizado em [-1, 1].

    def atualizar(self, alvo_controlador: float) -> float:
        """Aproxima a ativação do alvo do controlador e devolve o novo valor."""
        alvo = clamp(alvo_controlador, -1.0, 1.0)
        self.ativacao = lerp(self.ativacao, alvo, self.suavizacao)
        return self.ativacao

    def reset(self) -> None:
        self.ativacao = 0.0
