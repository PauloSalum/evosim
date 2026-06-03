"""Menu interativo do EvoSim — uso amigável, sem decorar parâmetros.

Lança um assistente de texto que guia o usuário por todos os modos (evoluir,
assistir, corrida, caça-caçador) com opções numeradas e valores padrão. Rode:

    python -m evosim            # ou:  python -m evosim.cli menu

Funciona apenas com a biblioteca padrão.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

from .aptidao.funcoes import listar_fitness
from .config import ConfigAmbiente, ConfigSimulacao
from .criaturas.presets import listar_presets
from .persistencia.serializacao import carregar

PASTA_SAVES = "runs"

_ALGORITMOS = [("es", "Estratégia Evolutiva (recomendado)"),
               ("ga", "Algoritmo Genético clássico")]
_CONTROLADORES = [("cpg", "Gerador de Padrão Central — aprende marcha rápido"),
                  ("mlp", "Rede Neural feedforward")]


# ---------------------------------------------------------------------------
# Primitivas de entrada (tolerantes a Enter = padrão, e a EOF).
# ---------------------------------------------------------------------------
def _ler(prompt: str, default: Optional[str] = None) -> Optional[str]:
    # EOFError (stdin fechado / Ctrl-D) propaga para encerrar o menu com
    # segurança, em vez de devolver o padrão em laço. Enter vazio usa o padrão.
    suf = f" [{default}]" if default not in (None, "") else ""
    s = input(f"{prompt}{suf}: ").strip()
    return s if s else default


def _escolher(titulo: str, rotulos: Sequence[str], default_idx: int = 0) -> int:
    print(f"\n{titulo}")
    for i, r in enumerate(rotulos, 1):
        marca = "  ◀ padrão" if i - 1 == default_idx else ""
        print(f"   {i}) {r}{marca}")
    while True:
        s = _ler("  Opção", str(default_idx + 1))
        if s is None:
            return default_idx
        if s.isdigit() and 1 <= int(s) <= len(rotulos):
            return int(s) - 1
        print("   ⚠ opção inválida, tente de novo.")


def _ler_int(prompt: str, default: int, minimo: int = 1) -> int:
    while True:
        s = _ler(prompt, str(default))
        try:
            v = int(s)
            if v < minimo:
                raise ValueError
            return v
        except (TypeError, ValueError):
            print(f"   ⚠ informe um inteiro ≥ {minimo}.")


def _ler_float(prompt: str, default: float, minimo: float = 0.0) -> float:
    while True:
        s = _ler(prompt, str(default))
        try:
            v = float(s)
            if v < minimo:
                raise ValueError
            return v
        except (TypeError, ValueError):
            print(f"   ⚠ informe um número ≥ {minimo}.")


def _sim_nao(prompt: str, default: bool = True) -> bool:
    s = _ler(f"{prompt} (s/n)", "s" if default else "n")
    return (s or "").lower().startswith("s")


def _saves_disponiveis() -> List[str]:
    achados: List[str] = []
    for base in (PASTA_SAVES, "configs"):
        for ext in ("json", "yaml", "yml"):
            achados.extend(sorted(glob.glob(os.path.join(base, f"*.{ext}"))))
    return achados


def _selecionar_save(titulo: str) -> Optional[str]:
    saves = _saves_disponiveis()
    if not saves:
        print(f"\n   (nenhum save encontrado em {PASTA_SAVES}/ ou configs/)")
        caminho = _ler("   Caminho do save")
        return caminho or None
    rotulos = saves + ["Digitar outro caminho..."]
    idx = _escolher(titulo, rotulos)
    if idx == len(saves):
        return _ler("   Caminho do save") or None
    return saves[idx]


# ---------------------------------------------------------------------------
# Telas dos modos.
# ---------------------------------------------------------------------------
def _stats_cb(ger: int, s: dict) -> None:
    if "melhor" in s:
        print(f"   ger {ger:3d} | melhor {s['melhor']:8.3f} | média {s['media']:8.3f}")
    else:
        print(f"   ger {ger:3d} | caçador {s['cacador_melhor']:7.2f} | "
              f"presa {s['presa_melhor']:7.2f}")


def tela_evoluir() -> None:
    from .modos.evolucao_isolada import rodar_evolucao_isolada
    from .persistencia.serializacao import salvar

    presets = listar_presets()
    preset = presets[_escolher("Qual criatura evoluir?", presets)]
    fits = listar_fitness()
    fitness = fits[_escolher("Função de aptidão (o que otimizar)?",
                             ["velocidade — andar/correr o mais longe",
                              "eficiência — distância por energia gasta",
                              "estabilidade — manter-se em pé e equilibrado"][:len(fits)])]
    algo = _ALGORITMOS[_escolher("Algoritmo evolutivo?",
                                 [r for _, r in _ALGORITMOS])][0]
    ctrl = _CONTROLADORES[_escolher("Tipo de cérebro?",
                                    [r for _, r in _CONTROLADORES])][0]
    geracoes = _ler_int("Número de gerações", 30)
    pop = _ler_int("Tamanho da população", 32)
    duracao = _ler_float("Duração de cada teste (segundos simulados)", 10.0, 1.0)
    seed = _ler_int("Semente aleatória (reprodutibilidade)", 1234, 0)
    n_workers = 1
    n_cpus = os.cpu_count() or 1
    if n_cpus > 1 and _sim_nao(f"Usar todos os {n_cpus} núcleos da CPU (treino mais rápido)?", True):
        n_workers = 0  # 0 => todos os núcleos

    # Assistir em 3D ENQUANTO treina (mostra o campeão de cada geração).
    mon_obj = None
    monitor = None
    if _sim_nao("Assistir em 3D durante o treino? (precisa de matplotlib)", False):
        intervalo = _ler_int("Mostrar a cada quantas gerações?", 1)
        try:
            from .render.matplotlib_view import Monitor3D
            mon_obj = Monitor3D(cada_geracao=intervalo)
            monitor = mon_obj.callback
        except Exception as e:
            print(f"   ⚠ não foi possível abrir a janela 3D ({e}). "
                  "Sigo o treino sem visualização.")

    print(f"\n▶ Evoluindo '{preset}' | fitness={fitness} | algo={algo} | "
          f"cérebro={ctrl} | {geracoes} gerações × {pop} indivíduos\n")
    sim = ConfigSimulacao(seed=seed, duracao_segundos=duracao)
    save = rodar_evolucao_isolada(
        preset, geracoes=geracoes, sim=sim, fitness=fitness, algoritmo=algo,
        tipo_controlador=ctrl, tamanho_pop=pop, seed=seed, callback=_stats_cb,
        monitor=monitor, n_workers=n_workers,
    )
    if mon_obj is not None:
        mon_obj.fechar(manter_aberto=False)
    melhor = save.melhores[0]["fitness"]
    print(f"\n✔ Concluído! Melhor fitness = {melhor:.3f}")

    padrao = os.path.join(PASTA_SAVES, f"{preset}.json")
    if _sim_nao(f"Salvar resultado em '{padrao}'?", True):
        caminho = _ler("Caminho do arquivo", padrao) or padrao
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        salvar(save, caminho)
        print(f"   💾 salvo em {caminho}")
        if _sim_nao("Assistir o vencedor agora (ASCII)?", True):
            _assistir(caminho)


def tela_continuar() -> None:
    from .modos.evolucao_isolada import continuar_evolucao
    from .persistencia.serializacao import salvar

    caminho = _selecionar_save("Continuar a evolução de qual save?")
    if not caminho:
        return
    save = carregar(caminho)
    print(f"   preset: {save.preset} | fitness original: {save.fitness_nome}")
    geracoes = _ler_int("Número de gerações", 30)
    pop = _ler_int("Tamanho da população", 32)
    duracao = _ler_float("Duração de cada teste (s)", 10.0, 1.0)
    seed = _ler_int("Semente", 1234, 0)

    # gravidade opcional (ex.: continuar o mesmo bicho na Lua).
    ambiente = None
    gy = _ler("Gravidade Y (Enter = manter a do save)", "")
    if gy:
        base = dict(save.ambiente) if save.ambiente else ConfigAmbiente().__dict__
        g = list(base.get("gravidade", [0.0, -9.81, 0.0])); g[1] = float(gy)
        validos = ConfigAmbiente().__dict__.keys()
        ambiente = ConfigAmbiente(**{k: v for k, v in {**base, "gravidade": g}.items()
                                     if k in validos})

    n_workers = 1
    n_cpus = os.cpu_count() or 1
    if n_cpus > 1 and _sim_nao(f"Usar todos os {n_cpus} núcleos da CPU?", True):
        n_workers = 0

    mon_obj = None
    monitor = None
    if _sim_nao("Assistir em 3D durante o treino? (precisa de matplotlib)", False):
        try:
            from .render.matplotlib_view import Monitor3D
            mon_obj = Monitor3D()
            monitor = mon_obj.callback
        except Exception as e:
            print(f"   ⚠ {e}")

    print(f"\n▶ Continuando '{save.preset}' por {geracoes} gerações (warm-start)\n")
    sim = ConfigSimulacao(seed=seed, duracao_segundos=duracao)
    novo = continuar_evolucao(
        save, geracoes=geracoes, ambiente=ambiente, sim=sim, tamanho_pop=pop,
        seed=seed, callback=_stats_cb, monitor=monitor, n_workers=n_workers,
    )
    if mon_obj is not None:
        mon_obj.fechar(manter_aberto=False)
    print(f"\n✔ Concluído! Melhor fitness = {novo.melhores[0]['fitness']:.3f}")
    if _sim_nao(f"Salvar (sobrescrever {caminho})?", True):
        dest = _ler("Caminho do arquivo", caminho) or caminho
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        salvar(novo, dest)
        print(f"   💾 salvo em {dest}")
        if _sim_nao("Assistir o vencedor agora (ASCII)?", True):
            _assistir(dest)


def _assistir(caminho: str, plano: str = "xy") -> None:
    from .criaturas.criatura import Criatura
    from .fisica.motor_interno import MotorInterno
    from .mathutils import Vec3
    from .render.ascii import assistir_terminal

    save = carregar(caminho)
    amb = ConfigAmbiente(**save.ambiente) if save.ambiente else ConfigAmbiente()
    sim = ConfigSimulacao()
    motor = MotorInterno(amb)
    motor.substeps = sim.substeps
    dna = save.dna_obj()
    cri = Criatura(dna, motor.construir_criatura(dna, Vec3(0, 0, 0)))
    cri.controlador = save.melhor_genoma().instanciar_controlador()
    for passo in range(sim.passos_totais):
        cri.passo_controle(motor.tempo)
        motor.passo(sim.dt)
        assistir_terminal(motor, passo, plano=plano)


def tela_assistir() -> None:
    caminho = _selecionar_save("Qual save assistir?")
    if not caminho:
        return
    planos = ["xy (lateral)", "xz (de cima)", "zy (de frente)"]
    plano = ["xy", "xz", "zy"][_escolher("Ângulo de câmera?", planos)]
    _assistir(caminho, plano)


def tela_assistir_3d() -> None:
    caminho = _selecionar_save("Qual save assistir em 3D?")
    if not caminho:
        return
    gif = None
    if _sim_nao("Exportar também um GIF?", False):
        gif = _ler("Caminho do GIF", "evolucao.gif") or "evolucao.gif"
    try:
        from .render.matplotlib_view import reproduzir_save
        print("   abrindo janela 3D (feche a janela para voltar)...")
        reproduzir_save(carregar(caminho), gif=gif, mostrar=True)
    except Exception as e:
        print(f"   ⚠ {e}")


def tela_corrida() -> None:
    from .modos.corrida import Competidor, rodar_corrida
    from .render.ascii import assistir_terminal

    saves = _saves_disponiveis()
    if len(saves) < 2:
        print("\n   ⚠ É preciso ao menos 2 saves para uma corrida. "
              "Evolua e salve algumas criaturas primeiro.")
        return
    print("\nDigite os números dos competidores separados por espaço "
          "(ex.: 1 3 4):")
    for i, s in enumerate(saves, 1):
        print(f"   {i}) {s}")
    escolha = _ler("  Competidores", "1 2") or "1 2"
    idxs = [int(x) - 1 for x in escolha.split() if x.isdigit() and 1 <= int(x) <= len(saves)]
    if len(idxs) < 2:
        print("   ⚠ seleção insuficiente.")
        return
    comps = [Competidor.de_save(carregar(saves[i])) for i in idxs]
    render = _sim_nao("Mostrar a corrida no terminal?", True)
    on_step = assistir_terminal if render else None
    ranking = rodar_corrida(comps, on_step=on_step)
    print("\n=== Resultado da corrida ===")
    for pos, (nome, dist) in enumerate(ranking, 1):
        print(f"   {pos}º  {nome:26s} {dist:7.2f} m")


def tela_caca() -> None:
    from .evolucao import criar_algoritmo
    from .criaturas.presets import criar_preset
    from .modos.caca_cacador import CoEvolucao
    from .persistencia.serializacao import salvar

    presets = listar_presets()
    cacador = presets[_escolher("Quem é o CAÇADOR?", presets, 1)]
    presa = presets[_escolher("Quem é a PRESA?", presets, 0)]
    geracoes = _ler_int("Número de gerações", 15)
    pop = _ler_int("Tamanho da população (por espécie)", 24)
    duracao = _ler_float("Duração de cada disputa (s)", 10.0, 1.0)
    seed = _ler_int("Semente aleatória", 1234, 0)

    print(f"\n▶ Co-evoluindo caçador='{cacador}' vs presa='{presa}'\n")
    amb = ConfigAmbiente()
    sim = ConfigSimulacao(seed=seed, duracao_segundos=duracao)
    factory = criar_algoritmo("es", tamanho_pop=pop, seed=seed)
    co = CoEvolucao(criar_preset(cacador), criar_preset(presa), amb, sim, factory)
    save_c, save_p = co.evoluir(geracoes, callback=_stats_cb)
    print("\n✔ Co-evolução concluída.")
    if _sim_nao("Salvar caçador e presa?", True):
        os.makedirs(PASTA_SAVES, exist_ok=True)
        pc = os.path.join(PASTA_SAVES, f"cacador_{cacador}.json")
        pp = os.path.join(PASTA_SAVES, f"presa_{presa}.json")
        salvar(save_c, pc); salvar(save_p, pp)
        print(f"   💾 {pc}\n   💾 {pp}")


def tela_web() -> None:
    porta = _ler_int("Porta do servidor", 8000, 1)
    print(f"\n   Abrindo a interface web em http://127.0.0.1:{porta}")
    print("   (feche com Ctrl+C para voltar ao terminal)")
    from .web import iniciar_servidor
    iniciar_servidor(port=porta)


def tela_listar() -> None:
    print("\nPresets:", ", ".join(listar_presets()))
    print("Fitness:", ", ".join(listar_fitness()))
    print("Algoritmos: es (estratégia evolutiva), ga (genético)")
    print("Controladores: cpg, mlp")


# ---------------------------------------------------------------------------
def executar() -> None:
    print("=" * 60)
    print("  EvoSim — Evolução de Locomoção Biológica em 3D")
    print("=" * 60)
    acoes = [
        ("Abrir interface WEB (3D no navegador) ★", tela_web),
        ("Evoluir uma nova criatura", tela_evoluir),
        ("Continuar evolução de um save (warm-start)", tela_continuar),
        ("Assistir um save (ASCII, no terminal)", tela_assistir),
        ("Assistir um save em 3D (matplotlib)", tela_assistir_3d),
        ("Corrida entre saves (sandbox)", tela_corrida),
        ("Caça e Caçador (co-evolução)", tela_caca),
        ("Listar opções disponíveis", tela_listar),
        ("Sair", None),
    ]
    while True:
        try:
            idx = _escolher("O que você quer fazer?", [r for r, _ in acoes])
            rotulo, fn = acoes[idx]
            if fn is None:
                print("\nAté logo! 👋")
                return
            fn()
            input("\n   [Enter] para voltar ao menu...")
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo! 👋")
            return
        except Exception as e:  # mantém o menu vivo após um erro inesperado
            print(f"\n   ⚠ erro: {e}")


if __name__ == "__main__":
    executar()
