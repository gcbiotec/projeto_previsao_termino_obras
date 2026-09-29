# Projeto de Previsão de Término de Obras

Estrutura inicial pronta para abrir no VS Code. A solução prevê dois cenários:

1. **Obra nova**: estima a duração total e o risco de ultrapassar o prazo inicial.
2. **Obra em andamento**: estima os dias restantes, a data provável de conclusão e o risco de ultrapassar o prazo vigente.

## 1. Como abrir no VS Code

1. Extraia o arquivo ZIP.
2. No VS Code, use **Arquivo > Abrir Pasta**.
3. Selecione a pasta `projeto_previsao_termino_obras`.
4. Abra o terminal integrado.
5. Execute os comandos abaixo.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Preparar os dados

Copie os arquivos de modelo da pasta `data/modelos_entrada` para `data/raw` e preencha com dados reais:

- `obras_novas.csv`
- `snapshots_obras_andamento.csv`

Os arquivos fornecidos contêm somente os cabeçalhos, sem dados simulados.

## 3. Executar

```bash
python -m src.main
```

O processo realizará:

- validação dos arquivos de entrada;
- criação das variáveis derivadas;
- separação entre treino e teste por obra;
- treinamento dos modelos de regressão e classificação;
- gravação dos modelos em `artifacts/models`;
- geração das previsões em `data/outputs`;
- gravação das métricas em `artifacts/metrics`.

## 4. Saídas esperadas

### Obras novas

- duração total prevista;
- data provável de conclusão;
- dias adicionais em relação ao prazo inicial;
- probabilidade de ultrapassar o prazo inicial;
- faixa de risco.

### Obras em andamento

- dias restantes previstos;
- data provável de conclusão;
- dias adicionais em relação ao prazo vigente;
- probabilidade de atraso;
- faixa provável de conclusão de 80%;
- faixa de risco.

## 5. Observação

Esta estrutura é um ponto de partida. Os nomes genéricos das colunas devem ser associados às colunas físicas das tabelas do SGA antes do treinamento real.
