import pandas as pd
import numpy as np

def clasificar_parcelas(df):
    df = df.copy()
    
    # Limpieza
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    
    # Condiciones
    condiciones = [
        (df["nitrogeno"] < 20) | (df["humedad"] < 25),
        (df["temperatura"] >= 35) | (df["dias_ultimo_riego"] >= 7)
    ]
    
    categorias = ["urgente", "moderada"]
    
    # Clasificación
    df["intervencion"] = np.select(
        condiciones,
        categorias,
        default="sin_intervencion"
    )
    
    # Orden final
    df = df.sort_values("dias_ultimo_riego").reset_index(drop=True)
    
    return df
