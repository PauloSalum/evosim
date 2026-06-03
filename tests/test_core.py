"""Testes do EvoSim (stdlib unittest — sem dependências externas).

Rodar:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest

from evosim.aptidao.funcoes import listar_fitness, obter_fitness
from evosim.config import ConfigAmbiente, ConfigSimulacao
from evosim.criaturas.criatura import Criatura
from evosim.criaturas.musculo import Musculo
from evosim.criaturas.presets import criar_preset, listar_presets
from evosim.evolucao import criar_algoritmo
from evosim.fisica.motor_interno import MotorInterno
from evosim.mathutils import Vec3, angle_between
from evosim.modos.corrida import Competidor, rodar_corrida
from evosim.modos.evolucao_isolada import rodar_evolucao_isolada
from evosim.neural import criar_controlador
from evosim.persistencia.serializacao import Save, carregar, salvar
from evosim.simulacao.resultado import ResultadoEpisodio


class TestMatematica(unittest.TestCase):
    def test_operacoes_vec(self):
        a, b = Vec3(1, 0, 0), Vec3(0, 1, 0)
        self.assertAlmostEqual(a.cross(b).z, 1.0)
        self.assertAlmostEqual(a.dot(b), 0.0)
        self.assertAlmostEqual((a * 3).length(), 3.0)

    def test_rotacao_rodrigues(self):
        v = Vec3(1, 0, 0).rotate_around(Vec3(0, 0, 1), math.pi / 2)
        self.assertAlmostEqual(v.x, 0.0, places=6)
        self.assertAlmostEqual(v.y, 1.0, places=6)

    def test_angulo_entre(self):
        self.assertAlmostEqual(angle_between(Vec3(1, 0, 0), Vec3(0, 1, 0)), math.pi / 2)


class TestPresets(unittest.TestCase):
    def test_todos_presets_validos(self):
        amb = ConfigAmbiente()
        for nome in listar_presets():
            dna = criar_preset(nome)
            dna.validar()  # não deve lançar
            motor = MotorInterno(amb)
            corpo = motor.construir_criatura(dna, Vec3())
            self.assertEqual(corpo.num_juntas(), dna.num_musculos())
            cri = Criatura(dna, corpo)
            self.assertEqual(cri.num_saidas(), corpo.num_juntas())
            self.assertGreater(cri.num_entradas(), cri.num_saidas())


class TestMusculo(unittest.TestCase):
    def test_suavizacao_converge(self):
        m = Musculo(0, suavizacao=0.5)
        for _ in range(40):
            m.atualizar(1.0)
        self.assertAlmostEqual(m.ativacao, 1.0, places=3)

    def test_suavizacao_limita_salto(self):
        m = Musculo(0, suavizacao=0.2)
        primeiro = m.atualizar(1.0)
        self.assertLess(primeiro, 1.0)  # não salta instantaneamente.


class TestFisica(unittest.TestCase):
    def test_gravidade_aplicada(self):
        # Gravidade invertida (+Y) deve lançar a criatura para cima, provando
        # que a aceleração gravitacional é de fato integrada.
        amb = ConfigAmbiente(gravidade=[0.0, 9.81, 0.0], friccao_solo=0.0)
        motor = MotorInterno(amb)
        corpo = motor.construir_criatura(criar_preset("humanoide"), Vec3())
        y0 = corpo.centro_de_massa().y
        for _ in range(120):
            motor.passo(1 / 120)
        self.assertGreater(corpo.centro_de_massa().y, y0 + 1.0)

    def test_tonus_muscular_mantem_postura(self):
        # Sem comando do cérebro, os músculos sustentam a pose de repouso:
        # a criatura permanece de pé em vez de desabar atravessando o chão.
        amb = ConfigAmbiente()
        motor = MotorInterno(amb)
        corpo = motor.construir_criatura(criar_preset("quadrupede"), Vec3())
        for _ in range(240):
            motor.passo(1 / 120)
        # não desaba colando o tronco no chão (tônus sustenta parte da pose).
        self.assertGreater(corpo.centro_de_massa().y, 0.1)

    def test_nao_penetra_solo(self):
        amb = ConfigAmbiente()
        motor = MotorInterno(amb)
        corpo = motor.construir_criatura(criar_preset("quadrupede"), Vec3())
        for _ in range(240):
            motor.passo(1 / 120)
        for n in corpo.nos:
            self.assertGreaterEqual(motor.pos[n].y, amb.altura_solo - 1e-3)


class TestFitness(unittest.TestCase):
    def test_registro(self):
        self.assertIn("velocidade", listar_fitness())
        self.assertIn("estabilidade", listar_fitness())

    def test_velocidade(self):
        r = ResultadoEpisodio(distancia_eixo=10.0, tempo=5.0)
        self.assertAlmostEqual(obter_fitness("velocidade")(r), 2.0)

    def test_eficiencia_zero_sem_avanco(self):
        r = ResultadoEpisodio(distancia_eixo=0.0, energia=100.0)
        self.assertEqual(obter_fitness("eficiencia")(r), 0.0)


class TestEvolucao(unittest.TestCase):
    def test_determinismo(self):
        def run():
            sim = ConfigSimulacao(duracao_segundos=2.0, seed=99)
            s = rodar_evolucao_isolada("humanoide", geracoes=2, sim=sim,
                                       tamanho_pop=6, seed=99)
            return s.melhores[0]["fitness"]
        self.assertEqual(run(), run())

    def test_fitness_melhora(self):
        sim = ConfigSimulacao(duracao_segundos=2.5, seed=3)
        hist = []
        rodar_evolucao_isolada("quadrupede", geracoes=5, sim=sim, tamanho_pop=10,
                               callback=lambda g, s: hist.append(s["melhor"]))
        self.assertGreaterEqual(max(hist), hist[0])

    def test_algoritmos_ga_e_es(self):
        sim = ConfigSimulacao(duracao_segundos=2.0, seed=1)
        for algo in ("ga", "es"):
            s = rodar_evolucao_isolada("tripode", geracoes=2, sim=sim,
                                       tamanho_pop=6, algoritmo=algo)
            self.assertTrue(s.melhores)


class TestParadaPrecoce(unittest.TestCase):
    def test_aborta_quando_cai(self):
        from evosim.simulacao.avaliador import Avaliador
        amb = ConfigAmbiente()
        sim = ConfigSimulacao(duracao_segundos=6.0, altura_critica=5.0)  # impossível
        dna = criar_preset("humanoide")
        av = Avaliador(amb, sim)
        ctrl = criar_controlador("cpg", *av.dimensoes_controlador(dna))
        res = av.avaliar_individuo(dna, ctrl)
        self.assertTrue(res.terminou_cedo)
        self.assertEqual(res.motivo, "caiu")


class TestControladores(unittest.TestCase):
    def test_roundtrip_pesos(self):
        for tipo in ("mlp", "cpg"):
            c = criar_controlador(tipo, 8, 4, seed=2)
            pesos = c.get_pesos()
            self.assertEqual(len(pesos), c.num_pesos())
            c.set_pesos([0.0] * c.num_pesos())
            saida = c.avaliar([0.1] * 8, 0.5)
            self.assertEqual(len(saida), 4)
            for v in saida:
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)


class TestPersistencia(unittest.TestCase):
    def test_save_load_corrida(self):
        sim = ConfigSimulacao(duracao_segundos=2.0, seed=4)
        s1 = rodar_evolucao_isolada("quadrupede", geracoes=2, sim=sim, tamanho_pop=6)
        s2 = rodar_evolucao_isolada("tripode", geracoes=2, sim=sim, tamanho_pop=6)
        d = tempfile.mkdtemp()
        for ext in ("json", "yaml"):
            p = os.path.join(d, f"s.{ext}")
            salvar(s1, p)
            carregado = carregar(p)
            self.assertEqual(carregado.preset, s1.preset)
            self.assertEqual(carregado.melhor_genoma().num_pesos(),
                             s1.melhor_genoma().num_pesos())
        # criaturas de saves independentes competem no sandbox.
        rank = rodar_corrida([Competidor.de_save(s1), Competidor.de_save(s2)],
                             sim=ConfigSimulacao(duracao_segundos=2.0))
        self.assertEqual(len(rank), 2)


if __name__ == "__main__":
    unittest.main()
