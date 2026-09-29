import pandas as pd
import pytest

from src.data.validation import validar_colunas


def test_validar_colunas_identifica_ausencia():
    df = pd.DataFrame({"ID_OBRA": [1]})
    with pytest.raises(ValueError):
        validar_colunas(df, ["ID_OBRA", "TIPO_OBRA"], "Teste")
