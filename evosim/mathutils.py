"""Matemática determinística em Python puro.

Todo o motor de física e os controladores neurais usam estas primitivas.
Não dependemos de numpy de propósito: operações em ``float`` nativo do
CPython são bit-a-bit reprodutíveis entre máquinas para a mesma sequência de
operações, o que é essencial para o requisito de determinismo.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

EPS = 1e-9


@dataclass(frozen=True)
class Vec3:
    """Vetor 3D imutável."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # --- operadores ---------------------------------------------------
    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    # --- geometria ----------------------------------------------------
    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length_sq(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalized(self) -> "Vec3":
        l = self.length()
        if l < EPS:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def distance(self, o: "Vec3") -> float:
        return (self - o).length()

    def rotate_around(self, axis: "Vec3", angle: float) -> "Vec3":
        """Rotação de Rodrigues em torno de ``axis`` (não precisa ser unitário)."""
        k = axis.normalized()
        if k.length_sq() < EPS:
            return self
        c = math.cos(angle)
        s = math.sin(angle)
        return (
            self * c
            + k.cross(self) * s
            + k * (k.dot(self) * (1.0 - c))
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z]


def vec(seq: Sequence[float]) -> Vec3:
    return Vec3(float(seq[0]), float(seq[1]), float(seq[2]))


def angle_between(a: Vec3, b: Vec3) -> float:
    """Ângulo (0..pi) entre dois vetores, numericamente estável."""
    cross = a.cross(b).length()
    dot = a.dot(b)
    return math.atan2(cross, dot)


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def make_rng(seed: int) -> random.Random:
    """RNG determinístico isolado (Mersenne Twister da stdlib)."""
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Funções de ativação para redes neurais / CPG.
# ---------------------------------------------------------------------------
def tanh(x: float) -> float:
    # math.tanh já satura em ~+-20 sem overflow.
    return math.tanh(x)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


ACTIVATIONS = {"tanh": tanh, "sigmoid": sigmoid, "relu": relu}


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
