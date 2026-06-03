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

```bash
# Núcleo (sem dependências). Opcional: pip install -r requirements.txt
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

Três níveis, do mais simples ao mais fiel:

```bash
# ASCII (qualquer terminal, sem dependências):
python -m evosim.cli assistir --save runs/humanoide.json --plano xy

# GIF 3D leve (matplotlib):  ver evosim/render/matplotlib_view.py
pip install matplotlib

# GUI 3D em tempo real (PyBullet):
pip install pybullet
python -c "from evosim.persistencia.serializacao import carregar; \
           from evosim.render.pybullet_view import assistir_save_3d; \
           assistir_save_3d(carregar('runs/humanoide.json'))"
```

O `--render` nos modos `corrida`/`caca` desenha a ação no terminal ao vivo.

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

### Rodar os testes
```bash
python -m unittest discover -s tests -v
```
