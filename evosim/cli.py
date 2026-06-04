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
import json
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
from .modos.evolucao_isolada import continuar_evolucao, rodar_evolucao_isolada
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


def cmd_scone(args) -> None:
    from . import scone
    if args.listar or (not scone.disponivel() and not args.reproduzir):
        print("Ambientes SCONE (sconegym):")
        for a in scone.AMBIENTES:
            print("  -", a)
        if not scone.disponivel():
            print("\n⚠ sconegym/SCONE não instalados nesta máquina. Veja docs/SCONE.md")
            print("  1) Instale o SCONE: https://scone.software (com Hyfydy ou OpenSim)")
            print("  2) git clone https://github.com/tgeijten/sconegym")
            print("     cd sconegym && pip install -r requirements.txt && pip install -e .")
        return
    if args.reproduzir:
        r = scone.reproduzir(args.reproduzir, env_id=(args.env or None), seed=args.seed)
        print(f"Episódio gravado (reward={r:.2f}). Abra no SCONE Studio para assistir.")
        return
    modo = f"continuando de {args.continuar}" if args.continuar else "do zero"
    print(f"Treinando política (CMA-ES) no {args.env} — {args.geracoes} gerações "
          f"× {args.pop} ({modo})")
    res = scone.treinar(
        env_id=args.env, geracoes=args.geracoes, tamanho_pop=args.pop,
        ocultas=args.ocultas, seed=args.seed, continuar_de=args.continuar,
        callback=lambda g, s: print(f"  ger {g:4d} | melhor {s['melhor']:9.2f} | "
                                    f"média {s['media']:9.2f}"),
    )
    if args.saida:
        os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
        with open(args.saida, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print(f"Política salva em {args.saida}. O melhor episódio foi gravado "
              f"em formato SCONE (.sto) — abra no SCONE Studio para assistir.")


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
        n_workers=args.workers, motor=args.motor,
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


def cmd_continuar(args) -> None:
    save = carregar(args.save)
    print(f"Continuando '{save.preset}' a partir de {args.save} "
          f"por {args.geracoes} gerações (warm-start)")
    sim = ConfigSimulacao(seed=args.seed, duracao_segundos=args.duracao)
    # opcionalmente troca a gravidade Y (mantém o resto do ambiente do save).
    ambiente = None
    if args.gravidade_y is not None:
        base = dict(save.ambiente) if save.ambiente else ConfigAmbiente().__dict__
        g = list(base.get("gravidade", [0.0, -9.81, 0.0]))
        g[1] = args.gravidade_y
        base = {**base, "gravidade": g}
        validos = ConfigAmbiente().__dict__.keys()
        ambiente = ConfigAmbiente(**{k: v for k, v in base.items() if k in validos})
    mon_obj = None
    monitor = None
    if args.assistir3d:
        from .render.matplotlib_view import Monitor3D
        mon_obj = Monitor3D(cada_geracao=args.cada_ger)
        monitor = mon_obj.callback
    novo = continuar_evolucao(
        save, geracoes=args.geracoes, ambiente=ambiente, sim=sim,
        fitness=(args.fitness or None), algoritmo=args.algoritmo,
        tamanho_pop=args.pop, seed=args.seed, callback=_print_stats,
        monitor=monitor, n_workers=args.workers, motor=args.motor,
    )
    if mon_obj is not None:
        mon_obj.fechar(manter_aberto=False)
    saida = args.saida or args.save  # por padrão, sobrescreve o próprio save.
    os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)
    salvar(novo, saida)
    print(f"Save escrito em {saida} (melhor fitness={novo.melhores[0]['fitness']:.3f})")
    if args.render:
        _assistir_ascii(novo)


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

    sc = sub.add_parser("scone",
                        help="Treina no sconegym (músculos Hill reais, estilo SCONE)")
    sc.add_argument("--env", default="sconewalk_h0918-v1")
    sc.add_argument("--geracoes", type=int, default=200)
    sc.add_argument("--pop", type=int, default=16)
    sc.add_argument("--ocultas", type=int, nargs="*", default=[64, 64])
    sc.add_argument("--seed", type=int, default=1234)
    sc.add_argument("--saida", default="")
    sc.add_argument("--continuar", default="", help="política .json para warm-start")
    sc.add_argument("--reproduzir", default="",
                    help="grava um episódio SCONE de uma política salva (para assistir)")
    sc.add_argument("--listar", action="store_true", help="lista os ambientes e sai")
    sc.set_defaults(func=cmd_scone)

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
    e.add_argument("--motor", default="auto", choices=["auto", "pybullet", "interno"],
                   help="motor de física (auto = pybullet se instalado)")
    e.set_defaults(func=cmd_evoluir)

    c = sub.add_parser("continuar",
                       help="Continua a evolução a partir de um save (warm-start)")
    c.add_argument("--save", required=True, help="save de partida")
    c.add_argument("--geracoes", type=int, default=20)
    c.add_argument("--fitness", default="", choices=[""] + listar_fitness(),
                   help="por padrão, mantém a fitness do save")
    c.add_argument("--algoritmo", default="es")
    c.add_argument("--pop", type=int, default=32)
    c.add_argument("--duracao", type=float, default=10.0)
    c.add_argument("--seed", type=int, default=1234)
    c.add_argument("--gravidade-y", dest="gravidade_y", type=float, default=None,
                   help="opcional: continua sob outra gravidade Y (ex.: -1.62 Lua)")
    c.add_argument("--saida", default="",
                   help="onde salvar (padrão: sobrescreve o próprio save)")
    c.add_argument("--render", action="store_true")
    c.add_argument("--assistir3d", action="store_true")
    c.add_argument("--cada-ger", dest="cada_ger", type=int, default=1)
    c.add_argument("--workers", type=int, default=1,
                   help="núcleos de CPU (0 = todos; 1 = serial)")
    c.add_argument("--motor", default="auto", choices=["auto", "pybullet", "interno"],
                   help="motor de física (auto = pybullet se instalado)")
    c.set_defaults(func=cmd_continuar)

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
