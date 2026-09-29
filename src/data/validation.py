"""Validações mínimas antes do treinamento."""

import pandas as pd


def validar_colunas(df: pd.DataFrame, obrigatorias: list[str], nome: str) -> None:
    faltantes = [coluna for coluna in obrigatorias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"{nome}: colunas ausentes: {faltantes}")

    if df.empty:
        raise ValueError(f"{nome}: o arquivo está vazio.")

    if df["ID_OBRA"].isna().any():
        raise ValueError(f"{nome}: existem linhas sem ID_OBRA.")


def validar_historico(
    df: pd.DataFrame,
    target_classificacao: str,
    minimo_obras: int = 20,
) -> None:
    quantidade = df["ID_OBRA"].nunique()
    if quantidade < minimo_obras:
        raise ValueError(
            f"O histórico contém {quantidade} obras. "
            f"O exemplo exige pelo menos {minimo_obras} para iniciar o piloto."
        )

    if df[target_classificacao].nunique() < 2:
        raise ValueError("O histórico precisa conter exemplos das duas classes.")
