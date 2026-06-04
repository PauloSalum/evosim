# Rota SCONE — músculos Hill reais e marcha natural

Esta é a rota para chegar perto dos resultados do **Thomas Geijtenbeek**
([SCONE](https://scone.software) / [Hyfydy](https://hyfydy.com)): modelos
**musculoesqueléticos** de verdade (ossos, juntas e **músculos Hill-type
roteados**), em vez do esqueleto de cápsulas do nosso motor. A nossa evolução
(CMA-ES + política MLP) otimiza o *reward* já no estilo SCONE; você **assiste no
SCONE Studio**, com os músculos coloridos por ativação (vermelho/azul).

> Por que não dá para mostrar no nosso viewer web? Porque os músculos existem
> só nos modelos do SCONE — o nosso motor (PyBullet) simula cápsulas rígidas.
> O SCONE Studio é o renderizador desses músculos.

---

## 1. Instalar (Windows)

1. **SCONE** — baixe e instale pelo site oficial: <https://scone.software/doku.php?id=download>.
   A instalação traz o **SCONE Studio** (o visualizador) e o binding Python
   **`sconepy`**. Por padrão usa o **OpenSim**; o **Hyfydy** (gratuito para uso
   acadêmico) deixa a simulação muito mais rápida e habilita todos os ambientes.

2. **sconepy no Python** — siga a página oficial
   [SconePy](https://scone.software/doku.php?id=sconepy) para deixar o `sconepy`
   visível ao seu Python (em geral é adicionar a pasta de instalação do SCONE ao
   `PYTHONPATH`, ou instalar o wheel que vem junto). Teste:
   ```bash
   python -c "import sconepy; print('sconepy ok')"
   ```

3. **sconegym**:
   ```bash
   git clone https://github.com/tgeijten/sconegym
   cd sconegym
   pip install -r requirements.txt
   pip install -e .
   ```
   Verifique:
   ```bash
   python -c "import sconegym, gym; print(gym.make('sconewalk_h0918-v1'))"
   ```

Use **Python 3.9** num ambiente virtual (o sconegym fixa `gym==0.13`), separado
do ambiente do EvoSim, se necessário.

---

## 2. Treinar com o EvoSim

```bash
# lista os ambientes (caminhar/correr; modelos h0918, h1622, h2190)
python -m evosim.cli scone --listar

# treina uma política (CMA-ES) no modelo de caminhada mais simples
python -m evosim.cli scone --env sconewalk_h0918-v1 --geracoes 300 --pop 16 \
    --saida runs/scone_h0918.json
```

Ao terminar, o **melhor episódio é gravado em formato SCONE** (`.sto`/`.par`) na
pasta de resultados do SCONE (normalmente `Documentos/SCONE/results/...`).

Ambientes disponíveis:

| ID                     | Modelo | O quê |
|------------------------|--------|-------|
| `sconewalk_h0918-v1`   | H0918  | caminhar (9 DOF, 18 músculos) — comece por aqui |
| `sconewalk_h1622-v1`   | H1622  | caminhar (16 DOF, 22 músculos) |
| `sconewalk_h2190-v1`   | H2190  | caminhar (21 DOF, 90 músculos) — o da sua imagem |
| `sconerun_h0918-v1` …  | …      | correr |

---

## 3. Assistir (SCONE Studio)

1. Abra o **SCONE Studio**.
2. `File → Open` e navegue até a pasta de resultados (o caminho é impresso ao
   fim do treino). Abra o `.par`/`.sto` do melhor.
3. Play — você verá o humanoide com os **músculos coloridos por ativação**.

Para gravar um episódio novo a partir de uma política salva (sem retreinar):
```bash
python -m evosim.cli scone --reproduzir runs/scone_h0918.json
```

---

## 4. Continuar um treino (warm-start)

```bash
python -m evosim.cli scone --env sconewalk_h0918-v1 \
    --continuar runs/scone_h0918.json --geracoes 300 --saida runs/scone_h0918.json
```

---

## Notas

* **Ação** = vetor de excitações musculares em `[0, 1]` (uma por músculo). A
  nossa MLP devolve `[-1, 1]` e mapeamos para `[0, 1]`.
* **Reward** = objetivo do SCONE (velocidade-alvo, suavidade, esforço/metabólico,
  limites de junta, autocolisão). É por isso que a marcha sai natural.
* CMA-ES de verdade (via uma biblioteca dedicada) tende a superar o nosso ES
  para muitos músculos; se quiser, dá para plugar o `cma` no lugar do nosso
  `EstrategiaEvolutiva` — a interface de genoma é a mesma.
* Implementação: [`evosim/scone.py`](../evosim/scone.py).
