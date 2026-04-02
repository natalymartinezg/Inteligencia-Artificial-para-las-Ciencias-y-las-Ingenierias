import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_filtrar_y_transformar_sensores():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función filtrar_y_transformar_sensores.
    """

    # 1. Configuración
    n = random.randint(5, 15)

    # 2. Generar datos (incluye negativos y posibles outliers)
    data = np.random.randn(n) * 20
    df = pd.DataFrame({'sensor': data})

    # Introducir NaNs (~10%)
    mask = np.random.choice([True, False], size=n, p=[0.1, 0.9])
    df.loc[mask, 'sensor'] = np.nan

    columna_sensor = 'sensor'

    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columna_sensor': columna_sensor
    }

    # ---------------------------------------------------------
    # 4. OUTPUT ESPERADO
    # ---------------------------------------------------------
    df_out = df.copy()

    # 1. Imputar con mediana
    mediana = df_out[columna_sensor].median()
    df_out[columna_sensor] = df_out[columna_sensor].fillna(mediana)

    # 2. Winsorización (percentil 95)
    p95 = np.percentile(df_out[columna_sensor], 95)
    df_out[columna_sensor] = np.where(
        df_out[columna_sensor] > p95,
        p95,
        df_out[columna_sensor]
    )

    # 3. Transformación log segura (evita NaN)
    df_out['sensor_log'] = np.log1p(np.maximum(df_out[columna_sensor], 0))

    return input_data, df_out


if __name__ == "__main__":
    i, o = generar_caso_de_uso_filtrar_y_transformar_sensores()

    print("---- inputs ----")
    for k, v in i.items():
        print("\n", k, ":\n", v)

    print("\n---- expected output ----\n", o)
