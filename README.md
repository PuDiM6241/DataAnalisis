# Regressão Linear e KNN

Atividade com Python, Pandas, Scikit-learn e Matplotlib.

## Objetivo

O objetivo da atividade é utilizar algoritmos de Machine Learning para realizar previsões a partir de dados:

* **Regressão Linear**
* **Regressão Linear Múltipla**
* **K-Nearest Neighbors (KNN)**

---

## lib utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Instalação

No terminal digite:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

# 1. Regressão Linear - Pressão e Temperatura

## Dataset

O arquivo utilizado é:

```text
data/aguaAlpes.csv
```

O dataset possui as seguintes variáveis:

| Variável   | Descrição                     |
| ---------- | ----------------------------- |
| `BPt`      | Temperatura de ebulição em °F |
| `Pressure` | Pressão em inHg               |

A temperatura de ebulição é utilizada para prever a pressão.

## Modelo

`LinearRegression`

## Previsões

O modelo realiza previsões para:

```text
Temperatura = 200°F
Temperatura = 215°F
```

## Visualização

Foi criado um gráfico contendo:

* eixo X: temperatura;
* eixo Y: pressão;
* pontos do dataset;
* linha de regressão.

---

# 2. Regressão Linear Múltipla

## Dataset

O arquivo utilizado é:

```text
data/notas.csv
```

| Variável | Descrição                        |
| -------- | -------------------------------- |
| `Livros` | Quantidade de livros lidos       |
| `Aulas`  | Quantidade de aulas participadas |
| `Nota`   | Nota                             |

As variáveis `Livros` e `Aulas` são utilizadas para prever a `Nota`.

## Modelo

`LinearRegression`:

A equação do modelo:

```text
Nota = coefLivros × Livros + coefAulas × Aulas + intercept
```

Onde:

* `coefLivros` representa a influência da quantidade de livros.
* `coefAulas` representa a influência da quantidade de aulas.
* `intercept` é o valor inicial da equação.

## Previsões

O modelo realiza previsões para diferentes combinações de livros e aulas:

```text
Livros = 2, Aulas = 11
Livros = 0, Aulas = 5
Livros = 4, Aulas = 20
Livros = 2, Aulas = 10
Livros = 4, Aulas = 15
```

## Visualização

Foi criado um gráfico 3D contendo:

* eixo X: quantidade de livros;
* eixo Y: quantidade de aulas;
* eixo Z: nota;
* pontos do dataset;
* plano de regressão.

O gráfico é salvo em:

```text
graphView/LivrosAulasNotasRalation.png
```

---

# 3. K-Nearest Neighbors (KNN)

## Dataset

O arquivo utilizado é:

```text
data/iris.csv
```

| Variável       | Descrição             |
| -------------- | --------------------- |
| `sepal_length` | Comprimento da sépala |
| `sepal_width`  | Largura da sépala     |
| `petal_length` | Comprimento da pétala |
| `petal_width`  | Largura da pétala     |
| `species`      | Espécie da flor       |

As quatro medidas são utilizadas para prever a espécie.

## Modelo

Foi utilizado o algoritmo KNN com 3 vizinhos:

O KNN compara uma nova flor com os exemplos existentes no dataset e utiliza os **3 vizinhos mais próximos** para determinar a espécie prevista.

## Previsões

Foram criados novos dados de flores para testar o modelo:

```text
sepal_length
sepal_width
petal_length
petal_width
```

O modelo então retorna a espécie prevista para cada uma delas.
```

---

# Estrutura do projeto

```text
.
├── data/
│   ├── aguaAlpes.csv
│   ├── notas.csv
│   └── iris.csv
│
├── graphView/
│   ├── LivrosAulasNotasRalation.png
│   └── BPtPressureRelation.png
│
├── AguaAlpes.py
├── LivroAulaNotas.py
└── IrisKNN.py
```

---

# Como executar

Execute os arquivos Python a partir da pasta principal do projeto:

```bash
python AguaAlpes.py
```

```bash
python LivroAulaNotas.py
```

```bash
python IrisKNN.py
```

Os resultados das previsões serão exibidos no terminal e os gráficos serão salvos na pasta `graphView`.

---
