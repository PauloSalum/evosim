# EvoSim — Evolução de Caminhada e Corrida Biológica em 3D

Sistema modular para **evoluir, comparar e co-evoluir** criaturas biomecânicas
em 3D. Criaturas são geradas proceduralmente a partir de um DNA, controladas por
redes neurais e otimizadas por algoritmos evolutivos sobre um motor de física de
passo de tempo **fixo e determinístico**.

> O núcleo roda **apenas com a biblioteca padrão do Python** (≥ 3.10) — sem
> numpy, sem compilação. PyBullet, PyYAML e matplotlib são extras opcionais.

---

## Índice
1. [Arquitetura de classes](#1-arquitetura-de-classes)
2. [Instalação e início rápido](#2-instalação-e-início-rápido)
3. [Loop principal (física + evolução)](#3-loop-principal-física--evolução)
4. [Modos de simulação](#4-modos-de-simulação)
5. [Arquivos de configuração (save/load)](#5-arquivos-de-configuração-saveload)
6. [Renderização](#6-renderização)
7. [Decisões de projeto](#7-decisões-de-projeto)

---

## 1. Arquitetura de classes

```
                         ┌────────────────────────┐
                         │      AmbienteFisico     │  (MotorFisica, abstrato)
                         │  passo fixo, gravidade, │
                         │  arrasto, solo          │
                         └───────────┬─────────────┘
              ┌──────────────────────┴───────────────────────┐
       MotorInterno (PBD, Python puro)            MotorPyBullet (corpos rígidos)
              │  constrói/instancia                          │
              ▼                                              ▼
        ┌───────────────┐   sensores    ┌──────────────────────────┐
        │   Criatura    │──────────────▶│    ControladorNeural     │
        │ (DNA + corpo) │◀──ativações───│  RedeNeuralMLP | CPG     │
        └──────┬────────┘   (músculos)  └──────────────────────────┘
               │ é descrita por
               ▼
        ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
        │  CriaturaDNA  │──▶│  SegmentoDNA  │──▶│   JuntaDNA    │
        │ (árvore)      │   │ (osso)        │   │ (+ Musculo)   │
        └───────────────┘   └───────────────┘   └───────────────┘

   Avaliador ── roda episódio (passo fixo + parada precoce) ──▶ ResultadoEpisodio
        │                                                            │
        ▼                                                            ▼
   AlgoritmoEvolutivo  (AlgoritmoGenetico | EstrategiaEvolutiva)   FuncaoAptidao
        │  opera sobre Genoma (vetor de pesos)                  (velocidade,
        ▼                                                        eficiencia,
   Executor ── laço de gerações ──▶ Save (JSON/YAML)            estabilidade)
```

| Classe | Módulo | Papel |
|---|---|---|
| `MotorFisica` / `CorpoCriatura` | `evosim.fisica.motor` | Interface abstrata do motor e handle uniforme da criatura |
| `MotorInterno` | `evosim.fisica.motor_interno` | Solver PBD determinístico em Python puro |
| `MotorPyBullet` | `evosim.fisica.motor_pybullet` | Adaptador opcional p/ corpos rígidos (PyBullet) |
| `CriaturaDNA`, `SegmentoDNA`, `JuntaDNA` | `evosim.criaturas.dna` | Morfologia serializável (árvore de ossos) |
| `Musculo` | `evosim.criaturas.musculo` | Atuador com suavização (LERP / inércia muscular) |
| `Criatura` | `evosim.criaturas.criatura` | Une corpo + músculos + cérebro; laço sensor→ação |
| `ControladorNeural`, `RedeNeuralMLP`, `GeradorPadraoCentral` | `evosim.neural` | "Cérebros" (MLP feedforward e CPG) |
| `FuncaoAptidao` | `evosim.aptidao.funcoes` | `velocidade`, `eficiencia`, `estabilidade` (extensível) |
| `Genoma` | `evosim.evolucao.genoma` | Unidade evolutiva (pesos + metadados) |
| `AlgoritmoEvolutivo`, `AlgoritmoGenetico`, `EstrategiaEvolutiva` | `evosim.evolucao.algoritmos` | GA clássico e (μ/μ_w, λ)-ES |
| `Avaliador`, `Executor` | `evosim.simulacao` | Loop de física+avaliação e laço de gerações |
| `Save` | `evosim.persistencia.serializacao` | Save/Load de genoma, DNA, ambiente e histórico |

---

## 2. Instalação e início rápido

### Interface WEB (3D no navegador) ★

A forma mais completa: visualizador 3D interativo + todos os controles no
navegador, **sem dependências** (o servidor é stdlib; o 3D é um renderizador
próprio em Canvas).

```bash
python -m evosim.cli web          # abre http://localhost:8000
```

Na web você pode:
* **Evoluir do zero** (escolhendo preset, fitness, algoritmo, cérebro, núcleos…);
* **Continuar a evolução a partir de um save** (warm-start) — "chegou no ponto,
  agora evolui outro / continua esse";
* **Mudar a gravidade (e atrito/arrasto) em tempo real** — os sliders
  re-simulam o melhor indivíduo na hora;
* **Assistir o campeão de cada geração em 3D enquanto treina**, com gráfico de
  fitness ao vivo; girar a câmera (arrastar) e dar zoom (roda do mouse);
* **Salvar** o resultado.

### Modo fácil — menu interativo (recomendado)

Sem decorar parâmetros: rode e escolha tudo por opções numeradas.

```bash
python -m evosim
```

```
============================================================
  EvoSim — Evolução de Locomoção Biológica em 3D
============================================================

O que você quer fazer?
   1) Evoluir uma nova criatura  ◀ padrão
   2) Assistir um save treinado
   3) Corrida entre saves (sandbox)
   4) Caça e Caçador (co-evolução)
   5) Listar opções disponíveis
   6) Sair
```

O menu guia preset, fitness, algoritmo, cérebro, gerações, etc. (Enter aceita o
padrão), salva o resultado e oferece assistir o vencedor.

### Modo avançado — linha de comando

```bash
python -m evosim.cli listar

# Evolui um humanoide por 30 gerações para velocidade máxima e salva o melhor:
python -m evosim.cli evoluir --preset humanoide --geracoes 30 \
    --fitness velocidade --algoritmo es --controlador cpg --saida runs/humanoide.json

# Assiste o vencedor no terminal (ASCII):
python -m evosim.cli assistir --save runs/humanoide.json
```

Presets disponíveis (reais e de fantasia):
`humanoide`, `quadrupede`, `hexapode`, `ave_bipede`, `centauro`, `voador_cauda`, `tripode`.

---

## 3. Loop principal (física + evolução)

O coração do sistema é um laço de **passo de tempo fixo** integrado à avaliação,
seleção e mutação. Versão condensada (ver `evosim/simulacao/`):

```python
from evosim import ConfigAmbiente, ConfigSimulacao
from evosim.criaturas.presets import criar_preset
from evosim.evolucao import criar_algoritmo
from evosim.simulacao import Executor

dna   = criar_preset("quadrupede")
amb   = ConfigAmbiente()                       # gravidade/arrasto/solo (imutáveis)
sim   = ConfigSimulacao(dt=1/120, seed=1234)   # passo FIXO + parada precoce
algo  = criar_algoritmo("es", tamanho_pop=32)  # fábrica do algoritmo evolutivo

exe = Executor(dna, amb, sim, algo, fitness_nome="velocidade", tipo_controlador="cpg")
save = exe.evoluir(geracoes=40, callback=lambda g, s: print(g, s["melhor"]))
```

Dentro de `Avaliador._rodar`, para cada indivíduo, a cada passo:

```python
criatura.passo_controle(motor.tempo)   # sensores → cérebro → músculos (suavizados)
motor.passo(dt)                        # física de passo FIXO (substeps PBD)
criatura.acumular_metricas(energia)    # energia, desvio vertical, oscilação

# Parada precoce (early termination):
if cm.y < sim.altura_critica:                       break  # caiu
if criatura.corpo.parte_proibida_no_solo():          break  # cabeça no chão
if t > aquecimento and avanco < deslocamento_minimo: break  # não andou
```

A evolução **só altera os pesos da rede** (e, opcionalmente, proporções dentro
dos limites do preset). Gravidade, arrasto e solo **nunca** mudam.

---

## 4. Modos de simulação

```bash
# (a) Evolução isolada — uma espécie evolui sozinha.
python -m evosim.cli evoluir --preset hexapode --geracoes 40 --saida runs/hex.json

# (b) Corrida (sandbox) — carrega SAVES INDEPENDENTES e os coloca para competir.
python -m evosim.cli corrida --saves runs/humanoide.json runs/hex.json --render

# (c) Caça e Caçador (co-evolução) — duas espécies, corrida armamentista.
python -m evosim.cli caca --cacador quadrupede --presa humanoide --geracoes 20 \
    --saida-cacador runs/cacador.json --saida-presa runs/presa.json
```

No modo caça, a cada passo cada criatura recebe a **posição relativa da outra**
como alvo dinâmico (entra no vetor de sensores via `definir_alvo`). O caçador
ganha fitness ao encostar na presa; a presa, sobrevivendo e mantendo distância.

---

## 5. Arquivos de configuração (save/load)

Exemplos prontos em [`configs/`](configs/):

* `environment.yaml` — parâmetros do mundo (imutáveis para a evolução).
* `simulacao.yaml` — passo fixo + critérios de parada precoce.
* `humanoide.dna.yaml` — morfologia completa de um preset.
* `exemplo_save.json` — save completo (DNA + ambiente + histórico + pesos).

**DNA de um segmento** (trecho de `humanoide.dna.yaml`):

```yaml
nome: humanoide
preset: humanoide
altura_inicial: 1.1
segmentos:
  - id: core
    papel: core
    direcao: [0, 1, 0]
    comprimento: 0.55
    massa: 8.0
  - id: perna_esq_coxa
    papel: coxa
    pai: core
    conecta_em: proximal        # liga-se à pélvis (extremidade proximal do tronco)
    direcao: [0.0, -1.0, 0.12]
    comprimento: 0.42
    massa: 2.2
    junta:                      # músculo/atuador desta junta
      tipo: dobradica
      eixo: [0, 0, 1]
      limite_min: -1.3
      limite_max: 1.3
      torque_max: 55.0          # teto de torque do "músculo"
      rigidez: 0.4
```

**Save / Load programático:**

```python
from evosim.persistencia.serializacao import salvar, carregar
salvar(save, "runs/humanoide.json")          # ou .yaml
save = carregar("runs/humanoide.json")
dna  = save.dna_obj()                          # CriaturaDNA
cerebro = save.melhor_genoma().instanciar_controlador()
```

Como cada save é autocontido (morfologia + ambiente + pesos), criaturas
treinadas **independentemente, em momentos diferentes**, podem ser carregadas
juntas no Modo Corrida ou Caça (`Competidor.de_save`).

---

## 6. Renderização

Todas as opções abaixo (exceto PyBullet) usam o **mesmo motor do treino**
(`MotorInterno`), então mostram o comportamento fiel ao que foi evoluído.

```bash
# 1) ASCII (qualquer terminal, sem dependências):
python -m evosim.cli assistir --save runs/humanoide.json --plano xy

# 2) 3D com matplotlib — assiste o melhor de um save (e opcional GIF):
pip install matplotlib pillow
python -m evosim.cli assistir3d --save runs/humanoide.json
python -m evosim.cli assistir3d --save runs/humanoide.json --gif h.gif --so-gif

# 3) Assistir em 3D ENQUANTO TREINA (campeão de cada geração, ao vivo):
python -m evosim.cli evoluir --preset humanoide --geracoes 30 --assistir3d --saida runs/h.json

# 4) GUI 3D interativa (PyBullet) — atenção: roda em OUTRO motor de física,
#    então o movimento difere do treino:
pip install pybullet
python -c "from evosim.persistencia.serializacao import carregar; \
           from evosim.render.pybullet_view import assistir_save_3d; \
           assistir_save_3d(carregar('runs/humanoide.json'))"
```

No menu interativo (`python -m evosim`), o fluxo "Evoluir" pergunta se você quer
**assistir em 3D durante o treino**, e há a opção "Assistir um save em 3D".
O `--render` nos modos `corrida`/`caca` desenha a ação no terminal ao vivo.

> Tudo roda em **CPU** (pure-Python, determinístico). Nada usa GPU.

---

## 7. Decisões de projeto

* **Motor de física abstrato.** Todo o sistema fala com `MotorFisica`/
  `CorpoCriatura`. `MotorInterno` (PBD, Python puro) garante execução
  determinística sem dependências; `MotorPyBullet` oferece corpos rígidos
  completos quando instalado. Trocar de um para o outro não exige mudar
  criaturas, cérebros nem a evolução.
* **Passo de tempo fixo.** A física avança em `dt` constante com subpassos,
  desacoplada do FPS de renderização → o mesmo genoma se comporta igual em
  qualquer máquina (verificado em `tests/test_core.py::test_determinismo`).
* **Músculos suavizados.** As saídas do cérebro passam por um filtro de
  ativação (LERP) antes de virar torque, simulando inércia/elasticidade e
  evitando tremores irreais.
* **Parâmetros do mundo imutáveis.** A evolução otimiza apenas o vetor de pesos
  do controlador, nunca as regras do ambiente.
* **Atuadores fisicamente plausíveis.** Músculos são forças *internas*: não
  deslocam o centro de massa (só gravidade e o solo, externos, deslocam) e têm
  velocidade angular limitada. Há ainda um teto para a velocidade de subida do
  corpo. Sem isso, a evolução "trapaceava" arremessando a criatura no ar.
* **Treino paralelo e determinístico.** A avaliação da população roda em
  múltiplos núcleos (`--workers 0` = todos), com resultado **idêntico** ao
  serial — a simulação não usa aleatoriedade; o RNG vive só na evolução.
  Em uma CPU de 4 núcleos, ~3,6× mais rápido.

### Acelerar o treino (multi-core)

```bash
# Usa todos os núcleos da CPU (mesmo resultado, muito mais rápido):
python -m evosim.cli evoluir --preset humanoide --geracoes 40 --workers 0 --saida runs/h.json
```

O menu interativo pergunta automaticamente se quer usar todos os núcleos.
Tudo roda em **CPU** (não usa GPU).

### Rodar os testes
```bash
python -m unittest discover -s tests -v
```
