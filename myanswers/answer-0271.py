import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def train_sepsis_detector(X, y, class_weight_ratio=10):

    # 1. División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 2. Modelo base
    rf = RandomForestClassifier(
        class_weight={0: 1, 1: class_weight_ratio},
        random_state=42
    )

    # 3. Búsqueda de hiperparámetros (igual al generador)
    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5]
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        scoring='recall',
        n_iter=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # 4. Evaluación
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return best_model, cm
