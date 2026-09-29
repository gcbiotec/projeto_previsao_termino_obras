"""Treinamento genérico dos modelos."""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from src.models.factory import criar_modelos
from src.data.validation import validar_historico


def treinar_conjunto(
    historico,
    features_numericas,
    features_categoricas,
    target_regressao,
    target_classificacao,
    prefixo,
    pasta_modelos,
    pasta_metricas,
):
    validar_historico(historico, target_classificacao)
    features = features_numericas + features_categoricas

    divisor = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    idx_treino, idx_teste = next(divisor.split(historico, groups=historico["ID_OBRA"]))
    treino = historico.iloc[idx_treino]
    teste = historico.iloc[idx_teste]

    regressao, classificacao, q10, q90 = criar_modelos(
        features_numericas, features_categoricas
    )

    regressao.fit(treino[features], treino[target_regressao])
    classificacao.fit(treino[features], treino[target_classificacao])
    q10.fit(treino[features], treino[target_regressao])
    q90.fit(treino[features], treino[target_regressao])

    previsto = regressao.predict(teste[features])
    probabilidade = classificacao.predict_proba(teste[features])[:, 1]
    inferior = q10.predict(teste[features])
    superior = q90.predict(teste[features])

    metricas = {
        "mae_dias": float(mean_absolute_error(teste[target_regressao], previsto)),
        "rmse_dias": float(mean_squared_error(teste[target_regressao], previsto) ** 0.5),
        "brier_score": float(brier_score_loss(teste[target_classificacao], probabilidade)),
        "cobertura_intervalo_80": float(np.mean(
            (teste[target_regressao] >= inferior) & (teste[target_regressao] <= superior)
        )),
    }
    if teste[target_classificacao].nunique() == 2:
        metricas["roc_auc"] = float(
            roc_auc_score(teste[target_classificacao], probabilidade)
        )

    pasta_modelos = Path(pasta_modelos)
    pasta_metricas = Path(pasta_metricas)
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_metricas.mkdir(parents=True, exist_ok=True)

    joblib.dump(regressao, pasta_modelos / f"{prefixo}_regressao.joblib")
    joblib.dump(classificacao, pasta_modelos / f"{prefixo}_classificacao.joblib")
    joblib.dump(q10, pasta_modelos / f"{prefixo}_q10.joblib")
    joblib.dump(q90, pasta_modelos / f"{prefixo}_q90.joblib")

    with open(pasta_metricas / f"{prefixo}_metricas.json", "w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    return metricas, (regressao, classificacao, q10, q90)
