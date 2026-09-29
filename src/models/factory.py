"""Fábrica dos pipelines de regressão, classificação e incerteza."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def criar_preprocessador(numericas, categoricas):
    numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numericas", numerico, numericas),
        ("categoricas", categorico, categoricas),
    ])


def criar_modelos(numericas, categoricas):
    def prep():
        return criar_preprocessador(numericas, categoricas)

    regressao = Pipeline([
        ("preparacao", prep()),
        ("modelo", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=20, random_state=42
        )),
    ])
    classificacao = Pipeline([
        ("preparacao", prep()),
        ("modelo", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )),
    ])
    q10 = Pipeline([
        ("preparacao", prep()),
        ("modelo", GradientBoostingRegressor(
            loss="quantile", alpha=0.10, n_estimators=200, random_state=42
        )),
    ])
    q90 = Pipeline([
        ("preparacao", prep()),
        ("modelo", GradientBoostingRegressor(
            loss="quantile", alpha=0.90, n_estimators=200, random_state=42
        )),
    ])
    return regressao, classificacao, q10, q90
