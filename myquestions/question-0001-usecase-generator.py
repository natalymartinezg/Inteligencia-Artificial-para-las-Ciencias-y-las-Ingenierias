import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import random

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso:
    - input: diccionario con argumentos para preparar_datos
    - output: resultado esperado (ground truth)
    """

    # =========================
    # 1. Dimensiones aleatorias
    # =========================
    n_rows = random.randint(5, 15)
    n_features = random.randint(2, 5)

    # =========================
    # 2. Crear DataFrame
    # =========================
    data = np.random.randn(n_rows, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)

    # =========================
    # 3. Asegurar NaNs
    # =========================
    n_nans = max(1, int(0.1 * df.size))
    for _ in range(n_nans):
        i = random.randint(0, n_rows - 1)
        j = random.randint(0, n_features - 1)
        df.iat[i, j] = np.nan

    # =========================
    # 4. Columna target
    # =========================
    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    # =========================
    # INPUT
    # =========================
    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }

    # =========================
    # OUTPUT ESPERADO
    # =========================
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputación
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    output_data = (X_scaled, y)

    return input_data, output_data


# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":

    input_data, output_data = generar_caso_de_uso_preparar_datos()

    print("=========== INPUT ===========")
    print("Target column:", input_data['target_col'])
    print("\nDataFrame:")
    print(input_data['df'])

    print("\nNaNs por columna:")
    print(input_data['df'].isna().sum())

    print("\n=========== OUTPUT ESPERADO ===========")
    X_out, y_out = output_data

    print("\nX (procesada):")
    print(X_out)

    print("\ny (target):")
    print(y_out)

    print("\nShapes:")
    print("X:", X_out.shape)
    print("y:", y_out.shape)
