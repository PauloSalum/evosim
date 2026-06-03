"""Interface abstrata do motor de física (`AmbienteFisico`).

A arquitetura desacopla totalmente o resto do sistema (criaturas, cérebros,
evolução) da engine de física concreta. Qualquer motor que implemente
``MotorFisica`` e devolva handles ``CorpoCriatura`` pode ser usado:

* ``MotorInterno``  — solver determinístico em Python puro (sem dependências).
* ``MotorPyBullet`` — adaptador para a engine de corpos rígidos PyBullet.

O contrato é desenhado em torno do que uma criatura precisa: ler sensores,
acionar músculos (alvo de ativação por junta) e consultar estado global.
"""
from __future__ import annotations

import abc
from typing import List, Optional, TYPE_CHECKING

from ..mathutils import Vec3

if TYPE_CHECKING:  # evita import circular em runtime
    from ..config import ConfigAmbiente
    from ..criaturas.dna import CriaturaDNA


class CorpoCriatura(abc.ABC):
    """Handle uniforme para uma criatura instanciada dentro de um motor.

    O atuador de cada junta recebe uma *ativação normalizada* em ``[-1, 1]``
    que o motor mapeia para um ângulo-alvo dentro dos limites da junta e
    persegue com torque limitado (modelo de músculo). Toda a suavização
    temporal (LERP/inércia muscular) acontece na camada ``Criatura``.
    """

    # --- juntas / músculos -------------------------------------------
    @abc.abstractmethod
    def num_juntas(self) -> int: ...

    @abc.abstractmethod
    def angulo_junta(self, i: int) -> float:
        """Ângulo atual da junta ``i`` em radianos (relativo ao repouso)."""

    @abc.abstractmethod
    def limites_junta(self, i: int) -> "tuple[float, float]": ...

    @abc.abstractmethod
    def definir_ativacao(self, i: int, valor_norm: float) -> None:
        """Define o alvo do músculo da junta ``i`` (``valor_norm`` em [-1,1])."""

    # --- estado global ------------------------------------------------
    @abc.abstractmethod
    def centro_de_massa(self) -> Vec3: ...

    @abc.abstractmethod
    def pos_core(self) -> Vec3:
        """Posição do tronco/pélvis."""

    @abc.abstractmethod
    def velocidade_cm(self) -> Vec3: ...

    @abc.abstractmethod
    def vetor_up_core(self) -> Vec3:
        """Eixo "para cima" do tronco — usado pelo giroscópio/equilíbrio."""

    # --- sensores de contato -----------------------------------------
    @abc.abstractmethod
    def contatos_pe(self) -> List[bool]:
        """Um booleano por segmento marcado como ``pe`` no DNA."""

    @abc.abstractmethod
    def parte_proibida_no_solo(self) -> bool:
        """True se um segmento proibido (ex.: cabeça) tocou o solo."""

    # --- energia ------------------------------------------------------
    @abc.abstractmethod
    def energia_acumulada(self) -> float:
        """Soma do |torque| aplicado por todos os músculos (custo metabólico)."""

    # --- competição (posição relativa a um alvo dinâmico) -------------
    def definir_alvo(self, alvo: Optional[Vec3]) -> None:
        self._alvo = alvo

    def vetor_para_alvo(self) -> Vec3:
        alvo = getattr(self, "_alvo", None)
        if alvo is None:
            return Vec3(0.0, 0.0, 0.0)
        return alvo - self.pos_core()


class MotorFisica(abc.ABC):
    """Engine de física com passo de tempo fixo."""

    def __init__(self, ambiente: "ConfigAmbiente") -> None:
        self.ambiente = ambiente
        self.tempo: float = 0.0

    @abc.abstractmethod
    def construir_criatura(
        self, dna: "CriaturaDNA", posicao_base: Vec3
    ) -> CorpoCriatura:
        """Instancia uma criatura a partir do DNA e devolve seu handle."""

    @abc.abstractmethod
    def passo(self, dt: float) -> None:
        """Avança a simulação por um passo fixo ``dt``."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Remove todas as criaturas e zera o tempo."""

    # Conveniência para renderização: snapshot de segmentos como linhas 3D.
    def coletar_segmentos_render(self) -> List["tuple[Vec3, Vec3, str]"]:
        return []
