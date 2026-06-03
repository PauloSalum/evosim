"""Interface de linha de comando do EvoSim.

Exemplos:
    python -m evosim.cli listar
    python -m evosim.cli evoluir --preset humanoide --geracoes 20 --saida runs/h.json
    python -m evosim.cli assistir --save runs/h.json
    python -m evosim.cli corrida --saves runs/h.json runs/q.json --render
    python -m evosim.cli caca --cacador quadrupede --presa humanoide --geracoes 15
"""
from __future__ import annotations

import argparse
import os
from typing import List

from .aptidao.funcoes import listar_fitness
from .config import ConfigAmbiente, ConfigSimulacao
from .criaturas.criatura import Criatura
from .criaturas.presets import criar_preset, listar_presets
from .evolucao import criar_algoritmo
from .fisica.motor_interno import MotorInterno
from .mathutils import Vec3
from .modos.caca_cacador import CoEvolucao
from .modos.corrida import Competidor, rodar_corrida
from .modos.evolucao_isolada import rodar_evolucao_isolada
from .persistencia.serializacao import carregar, salvar
from .render.ascii import assistir_terminal


def _print_stats(ger: int, stats: dict) -> None:
    if "melhor" in stats:
        print(f"  ger {ger:3d} | melhor {stats['melhor']:8.3f} | "
              f"media {stats['media']:8.3f} | pior {stats['pior']:8.3f}")
    else:
        print(f"  ger {ger:3d} | caçador {stats['cacador_melhor']:7.2f} | "
              f"presa {stats['presa_melhor']:7.2f}")


def cmd_menu(_args) -> None:
    from .menu import executar
    executar()


def cmd_web(args) -> None:
    from .web import iniciar_servidor
    iniciar_servidor(host=args.host, port=args.port,
                     abrir_navegador=not args.sem_navegador)


def cmd_listar(_args) -> None:
    print("Presets:", ", ".join(listar_presets()))
    print("Fitness:", ", ".join(listar_fitness()))
    print("Algoritmos: ga (genético), es (estratégia evolutiva)")
    print("Controladores: mlp, cpg")


def cmd_evoluir(args) -> None:
    print(f"Evoluindo '{args.preset}' por {args.geracoes} gerações "
          f"[fitness={args.fitness}, algo={args.algoritmo}, ctrl={args.controlador}]")
    sim = ConfigSimulacao(seed=args.seed, duracao_segundos=args.duracao)
    mon_obj = None
    monitor = None
    if args.assistir3d:  # janela 3D ao vivo com o campeão de cada geração.
        from .render.matplotlib_view import Monitor3D
        mon_obj = Monitor3D(cada_geracao=args.cada_ger)
        monitor = mon_obj.callback
    save = rodar_evolucao_isolada(
        args.preset, geracoes=args.geracoes, sim=sim,
        fitness=args.fitness, algoritmo=args.algoritmo,
        tipo_controlador=args.controlador, tamanho_pop=args.pop,
        seed=args.seed, callback=_print_stats, monitor=monitor,
        n_workers=args.workers,
    )
    if mon_obj is not None:
        mon_obj.fechar(manter_aberto=False)
    if args.saida:
        os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
        salvar(save, args.saida)
        print(f"Save escrito em {args.saida} "
              f"(melhor fitness={save.melhores[0]['fitness']:.3f})")
    if args.render:
        _assistir_ascii(save)


def cmd_assistir(args) -> None:
    save = carregar(args.save)
    _assistir_ascii(save, plano=args.plano)


def cmd_assistir3d(args) -> None:
    from .render.matplotlib_view import reproduzir_save
    reproduzir_save(carregar(args.save), gif=args.gif or None,
                    fps=args.fps, mostrar=not args.so_gif)


def _assistir_ascii(save, plano: str = "xy") -> None:
    ambiente = ConfigAmbiente(**save.ambiente) if save.ambiente else ConfigAmbiente()
    sim = ConfigSimulacao()
    motor = MotorInterno(ambiente); motor.substeps = sim.substeps
    dna = save.dna_obj()
    corpo = motor.construir_criatura(dna, Vec3(0, 0, 0))
    cri = Criatura(dna, corpo)
    cri.controlador = save.melhor_genoma().instanciar_controlador()
    for passo in range(sim.passos_totais):
        cri.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        assistir_terminal(motor, passo, plano=plano)


def cmd_corrida(args) -> None:
    comps: List[Competidor] = []
    for caminho in args.saves:
        comps.append(Competidor.de_save(carregar(caminho)))
    on_step = assistir_terminal if args.render else None
    ranking = rodar_corrida(comps, on_step=on_step)
    print("\n=== Resultado da corrida ===")
    for pos, (nome, dist) in enumerate(ranking, 1):
        print(f"  {pos}º  {nome:24s}  {dist:7.2f} m")


def cmd_caca(args) -> None:
    ambiente = ConfigAmbiente()
    sim = ConfigSimulacao(seed=args.seed, duracao_segundos=args.duracao)
    factory = criar_algoritmo(args.algoritmo, tamanho_pop=args.pop, seed=args.seed)
    co = CoEvolucao(criar_preset(args.cacador), criar_preset(args.presa),
                    ambiente, sim, factory, tipo_controlador=args.controlador)
    save_c, save_p = co.evoluir(args.geracoes, callback=_print_stats)
    if args.saida_cacador:
        salvar(save_c, args.saida_cacador)
        print(f"Caçador salvo em {args.saida_cacador}")
    if args.saida_presa:
        salvar(save_p, args.saida_presa)
        print(f"Presa salva em {args.saida_presa}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evosim",
        description="Simulação evolutiva 3D. Sem subcomando, abre o menu interativo.",
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("menu", help="Menu interativo (amigável)").set_defaults(func=cmd_menu)
    sub.add_parser("listar", help="Lista presets/fitness/algoritmos").set_defaults(func=cmd_listar)

    w = sub.add_parser("web", help="Interface web (3D no navegador)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--sem-navegador", dest="sem_navegador", action="store_true",
                   help="não abrir o navegador automaticamente")
    w.set_defaults(func=cmd_web)

    e = sub.add_parser("evoluir", help="Modo Evolução Isolada")
    e.add_argument("--preset", required=True, choices=listar_presets())
    e.add_argument("--geracoes", type=int, default=20)
    e.add_argument("--fitness", default="velocidade", choices=listar_fitness())
    e.add_argument("--algoritmo", default="es")
    e.add_argument("--controlador", default="cpg", choices=["mlp", "cpg"])
    e.add_argument("--pop", type=int, default=32)
    e.add_argument("--duracao", type=float, default=10.0)
    e.add_argument("--seed", type=int, default=1234)
    e.add_argument("--saida", default="")
    e.add_argument("--render", action="store_true",
                   help="assistir o vencedor em ASCII ao final")
    e.add_argument("--assistir3d", action="store_true",
                   help="janela 3D ao vivo do campeão a cada geração (matplotlib)")
    e.add_argument("--cada-ger", dest="cada_ger", type=int, default=1,
                   help="intervalo de gerações para atualizar a janela 3D")
    e.add_argument("--workers", type=int, default=1,
                   help="núcleos de CPU para o treino (0 = todos; 1 = serial)")
    e.set_defaults(func=cmd_evoluir)

    a = sub.add_parser("assistir", help="Reproduz um save em ASCII")
    a.add_argument("--save", required=True)
    a.add_argument("--plano", default="xy", choices=["xy", "xz", "zy"])
    a.set_defaults(func=cmd_assistir)

    a3 = sub.add_parser("assistir3d", help="Reproduz um save em 3D (matplotlib)")
    a3.add_argument("--save", required=True)
    a3.add_argument("--gif", default="", help="caminho para exportar um GIF")
    a3.add_argument("--fps", type=int, default=30)
    a3.add_argument("--so-gif", dest="so_gif", action="store_true",
                    help="apenas exporta o GIF, sem abrir janela")
    a3.set_defaults(func=cmd_assistir3d)

    c = sub.add_parser("corrida", help="Modo Corrida (Sandbox)")
    c.add_argument("--saves", nargs="+", required=True)
    c.add_argument("--render", action="store_true")
    c.set_defaults(func=cmd_corrida)

    k = sub.add_parser("caca", help="Modo Caça e Caçador (co-evolução)")
    k.add_argument("--cacador", required=True, choices=listar_presets())
    k.add_argument("--presa", required=True, choices=listar_presets())
    k.add_argument("--geracoes", type=int, default=15)
    k.add_argument("--algoritmo", default="es")
    k.add_argument("--controlador", default="cpg", choices=["mlp", "cpg"])
    k.add_argument("--pop", type=int, default=24)
    k.add_argument("--duracao", type=float, default=10.0)
    k.add_argument("--seed", type=int, default=1234)
    k.add_argument("--saida-cacador", dest="saida_cacador", default="")
    k.add_argument("--saida-presa", dest="saida_presa", default="")
    k.set_defaults(func=cmd_caca)
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):  # sem subcomando → menu interativo.
        from .menu import executar
        executar()
        return
    args.func(args)


if __name__ == "__main__":
    main()
