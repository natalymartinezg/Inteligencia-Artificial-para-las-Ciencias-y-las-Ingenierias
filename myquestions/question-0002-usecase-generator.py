import pandas as pd
import numpy as np
import random

def calcular_variacion_por_grupo():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_variacion_por_grupo.
    """

    n = random.randint(5, 15)

    df = pd.DataFrame({
        'grupo': np.random.choice(['A', 'B', 'C'], size=n),
        'valor': np.random.randint(0, 100, size=n)
    })

    col_grupo = 'grupo'
    col_valor = 'valor'

    input_data = {
        'df': df.copy(),
        'col_grupo': col_grupo,
        'col_valor': col_valor
    }

    df_out = df.copy()

    df_out = df_out.sort_values(by=[col_grupo]).reset_index(drop=True)
    df_out['variacion'] = df_out.groupby(col_grupo)[col_valor].diff()
    df_out['variacion'] = df_out['variacion'].fillna(0)

    return input_data, df_out


if __name__ == "__main__":
    i, o = calcular_variacion_por_grupo()

    print("---- inputs ----")
    for k, v in i.items():
        print("\n", k, ":\n", v)

    print("\n---- expected output ----\n", o)
