"""Leitura dos CSVs e preparação das bases analíticas."""

import pandas as pd

from src.config import COLUNAS_OBRA_NOVA, COLUNAS_ANDAMENTO
from src.data.validation import validar_colunas


def carregar_csv(caminho) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, encoding="utf-8")


def preparar_obras_novas(df: pd.DataFrame):
    validar_colunas(df, COLUNAS_OBRA_NOVA, "Obras novas")
    df = df.copy()

    for coluna in ["DATA_PREVISTA_INICIO", "DATA_CONCLUSAO_REAL", "DATA_FIM_INICIAL"]:
        df[coluna] = pd.to_datetime(df[coluna], errors="coerce", dayfirst=True)

    historico = df[df["DATA_CONCLUSAO_REAL"].notna()].copy()
    abertas = df[df["DATA_CONCLUSAO_REAL"].isna()].copy()

    historico["TARGET_DURACAO_TOTAL_DIAS"] = (
        historico["DATA_CONCLUSAO_REAL"] - historico["DATA_PREVISTA_INICIO"]
    ).dt.days
    historico["TARGET_ULTRAPASSOU_PRAZO"] = (
        historico["DATA_CONCLUSAO_REAL"] > historico["DATA_FIM_INICIAL"]
    ).astype(int)

    historico = historico[historico["TARGET_DURACAO_TOTAL_DIAS"] > 0]
    return historico, abertas


def preparar_obras_andamento(df: pd.DataFrame):
    validar_colunas(df, COLUNAS_ANDAMENTO, "Obras em andamento")
    df = df.copy()

    for coluna in [
        "DATA_CORTE", "DATA_INICIO_EFETIVA", "DATA_FIM_VIGENTE", "DATA_CONCLUSAO_REAL"
    ]:
        df[coluna] = pd.to_datetime(df[coluna], errors="coerce", dayfirst=True)

    df["DIAS_DECORRIDOS"] = (
        df["DATA_CORTE"] - df["DATA_INICIO_EFETIVA"]
    ).dt.days
    df["PERCENTUAL_PRAZO_CONSUMIDO"] = (
        100 * df["DIAS_DECORRIDOS"] / df["PRAZO_VIGENTE_DIAS"]
    )
    df["DESVIO_FISICO_TEMPORAL"] = (
        df["PERCENTUAL_EXECUCAO_FISICA"] - df["PERCENTUAL_PRAZO_CONSUMIDO"]
    )

    historico = df[df["DATA_CONCLUSAO_REAL"].notna()].copy()
    historico["TARGET_DIAS_RESTANTES"] = (
        historico["DATA_CONCLUSAO_REAL"] - historico["DATA_CORTE"]
    ).dt.days
    historico["TARGET_ULTRAPASSOU_PRAZO"] = (
        historico["DATA_CONCLUSAO_REAL"] > historico["DATA_FIM_VIGENTE"]
    ).astype(int)
    historico = historico[
        (historico["TARGET_DIAS_RESTANTES"] >= 0)
        & (historico["PRAZO_VIGENTE_DIAS"] > 0)
    ]

    abertas = df[df["DATA_CONCLUSAO_REAL"].isna()].copy()
    abertas = (
        abertas.sort_values("DATA_CORTE")
        .groupby("ID_OBRA", as_index=False)
        .tail(1)
    )
    return historico, abertas
