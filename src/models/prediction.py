"""Geração das saídas objetivas para os dois estágios da obra."""

import numpy as np
import pandas as pd


def faixa_risco(probabilidade):
    if probabilidade < 0.30:
        return "BAIXO"
    if probabilidade < 0.60:
        return "MODERADO"
    if probabilidade < 0.80:
        return "ALTO"
    return "CRITICO"


def prever_obras_novas(df, modelos, features):
    regressao, classificacao, q10, q90 = modelos
    X = df[features]
    duracao = np.maximum(0, regressao.predict(X))
    prob = classificacao.predict_proba(X)[:, 1]

    resultado = df[["ID_OBRA", "DATA_PREVISTA_INICIO", "DATA_FIM_INICIAL"]].copy()
    resultado["DURACAO_TOTAL_PREVISTA_DIAS"] = np.rint(duracao).astype(int)
    resultado["DATA_PREVISTA_MODELO"] = (
        resultado["DATA_PREVISTA_INICIO"]
        + pd.to_timedelta(resultado["DURACAO_TOTAL_PREVISTA_DIAS"], unit="D")
    )
    resultado["DIAS_ADICIONAIS_PREVISTOS"] = np.maximum(
        0, (resultado["DATA_PREVISTA_MODELO"] - resultado["DATA_FIM_INICIAL"]).dt.days
    )
    resultado["PROBABILIDADE_ATRASO_PCT"] = np.round(prob * 100, 1)
    resultado["RISCO_ATRASO"] = [faixa_risco(p) for p in prob]
    resultado["VERSAO_MODELO"] = "obra_nova_1.0"
    return resultado


def prever_obras_andamento(df, modelos, features):
    regressao, classificacao, q10, q90 = modelos
    X = df[features]
    dias = np.maximum(0, regressao.predict(X))
    inferior = np.maximum(0, q10.predict(X))
    superior = np.maximum(0, q90.predict(X))
    prob = classificacao.predict_proba(X)[:, 1]

    resultado = df[["ID_OBRA", "DATA_CORTE", "DATA_FIM_VIGENTE"]].copy()
    resultado["DIAS_RESTANTES_PREVISTOS"] = np.rint(dias).astype(int)
    resultado["DATA_PREVISTA_MODELO"] = (
        resultado["DATA_CORTE"]
        + pd.to_timedelta(resultado["DIAS_RESTANTES_PREVISTOS"], unit="D")
    )
    resultado["DIAS_ADICIONAIS_PREVISTOS"] = np.maximum(
        0, (resultado["DATA_PREVISTA_MODELO"] - resultado["DATA_FIM_VIGENTE"]).dt.days
    )
    resultado["PROBABILIDADE_ATRASO_PCT"] = np.round(prob * 100, 1)
    resultado["RISCO_ATRASO"] = [faixa_risco(p) for p in prob]
    resultado["DATA_LIMITE_INFERIOR_80"] = (
        resultado["DATA_CORTE"]
        + pd.to_timedelta(np.rint(np.minimum(inferior, superior)).astype(int), unit="D")
    )
    resultado["DATA_LIMITE_SUPERIOR_80"] = (
        resultado["DATA_CORTE"]
        + pd.to_timedelta(np.rint(np.maximum(inferior, superior)).astype(int), unit="D")
    )
    resultado["VERSAO_MODELO"] = "obra_andamento_1.0"
    return resultado
