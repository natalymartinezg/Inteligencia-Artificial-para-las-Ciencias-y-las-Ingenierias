import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

def aplicar_label_encoding():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función aplicar_label_encoding.
    """

    n = random.randint(5, 15)

    df = pd.DataFrame({
        'categoria': np.random.choice(['rojo', 'azul', 'verde', 'amarillo'], size=n)
    })

    columna = 'categoria'

    input_data = {
        'df': df.copy(),
        'columna': columna
    }

    df_out = df.copy()

    encoder = LabelEncoder()
    df_out[columna] = encoder.fit_transform(df_out[columna])

    return input_data, df_out


if __name__ == "__main__":
    i, o = aplicar_label_encoding()

    print("---- inputs ----")
    for k, v in i.items():
        print("\n", k, ":\n", v)

    print("\n---- expected output ----\n", o)
