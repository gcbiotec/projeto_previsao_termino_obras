"""Ponto de entrada do projeto."""

from src.config import (
    ARQUIVO_OBRAS_NOVAS,
    ARQUIVO_OBRAS_ANDAMENTO,
    DATA_OUTPUTS,
    MODELS_DIR,
    METRICS_DIR,
    FEATURES_NOVA_NUM,
    FEATURES_NOVA_CAT,
    FEATURES_ANDAMENTO_NUM,
    FEATURES_ANDAMENTO_CAT,
)
from src.data.preparation import (
    carregar_csv,
    preparar_obras_novas,
    preparar_obras_andamento,
)
from src.models.training import treinar_conjunto
from src.models.prediction import prever_obras_novas, prever_obras_andamento


def executar_obras_novas():
    historico, novas = preparar_obras_novas(carregar_csv(ARQUIVO_OBRAS_NOVAS))
    metricas, modelos = treinar_conjunto(
        historico,
        FEATURES_NOVA_NUM,
        FEATURES_NOVA_CAT,
        "TARGET_DURACAO_TOTAL_DIAS",
        "TARGET_ULTRAPASSOU_PRAZO",
        "obra_nova",
        MODELS_DIR,
        METRICS_DIR,
    )
    if not novas.empty:
        resultado = prever_obras_novas(
            novas, modelos, FEATURES_NOVA_NUM + FEATURES_NOVA_CAT
        )
        resultado.to_csv(
            DATA_OUTPUTS / "previsoes_obras_novas.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return metricas


def executar_obras_andamento():
    historico, abertas = preparar_obras_andamento(
        carregar_csv(ARQUIVO_OBRAS_ANDAMENTO)
    )
    metricas, modelos = treinar_conjunto(
        historico,
        FEATURES_ANDAMENTO_NUM,
        FEATURES_ANDAMENTO_CAT,
        "TARGET_DIAS_RESTANTES",
        "TARGET_ULTRAPASSOU_PRAZO",
        "obra_andamento",
        MODELS_DIR,
        METRICS_DIR,
    )
    if not abertas.empty:
        resultado = prever_obras_andamento(
            abertas, modelos, FEATURES_ANDAMENTO_NUM + FEATURES_ANDAMENTO_CAT
        )
        resultado.to_csv(
            DATA_OUTPUTS / "previsoes_obras_andamento.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return metricas


def main():
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Iniciando modelo para obras novas...")
    print(executar_obras_novas())

    print("\nIniciando modelo para obras em andamento...")
    print(executar_obras_andamento())

    print("\nProcesso concluído. Consulte data/outputs e artifacts/metrics.")


if __name__ == "__main__":
    main()
