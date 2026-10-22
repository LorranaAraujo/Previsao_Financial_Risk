"""
Módulo de Modelagem Preditiva.

Implementa três arquiteturas para previsão de volatilidade:
1. XGBoost Regressor
2. LightGBM Regressor
3. Rede Neural MLP (Perceptron Multicamadas via Keras / TensorFlow)
"""

from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    LIGHTGBM_PARAMS,
    NEURAL_NETWORK_PARAMS,
    RANDOM_STATE,
    XGBOOST_PARAMS,
)
from src.preprocessing import DatasetModelagem


def treinar_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any] = XGBOOST_PARAMS,
) -> Tuple[xgb.XGBRegressor, np.ndarray]:
    """
    Treina o modelo de Gradient Boosting XGBoost.
    """
    print("🔵 [1/3] Treinando XGBoost...")
    modelo = xgb.XGBRegressor(**params)
    modelo.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    predicoes = modelo.predict(X_val)
    print("  ✓ XGBoost concluído com sucesso!")
    return modelo, predicoes


def treinar_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any] = LIGHTGBM_PARAMS,
) -> Tuple[lgb.LGBMRegressor, np.ndarray]:
    """
    Treina o modelo LightGBM Regressor.
    """
    print("🟡 [2/3] Treinando LightGBM...")
    modelo = lgb.LGBMRegressor(**params)
    modelo.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )
    predicoes = modelo.predict(X_val)
    print("  ✓ LightGBM concluído com sucesso!")
    return modelo, predicoes


def construir_rede_neural(input_dim: int, lr: float = 1e-3) -> keras.Model:
    """
    Constrói a arquitetura MLP (Multilayer Perceptron) com Regularização:
    - Camadas densas com ativação ReLU
    - Batch Normalization
    - Dropout para controle de overfitting
    - Saída linear única para regressão
    """
    modelo = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),  # Saída contínua
        ],
        name="MLP_Previsao_Volatilidade",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return modelo


def treinar_rede_neural(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any] = NEURAL_NETWORK_PARAMS,
) -> Tuple[keras.Model, keras.callbacks.History, np.ndarray]:
    """
    Treina o modelo de Deep Learning MLP com Early Stopping.
    """
    print("🟣 [3/3] Treinando Rede Neural (MLP)...")
    tf.random.set_seed(params.get("random_state", RANDOM_STATE))

    input_dim = X_train.shape[1]
    modelo = construir_rede_neural(input_dim, lr=params.get("learning_rate", 1e-3))

    paciencia = params.get("early_stopping_patience", 20)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=paciencia,
        restore_best_weights=True,
    )

    history = modelo.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params.get("epochs", 200),
        batch_size=params.get("batch_size", 32),
        callbacks=[early_stop],
        verbose=0,
    )

    predicoes = modelo.predict(X_val, verbose=0).flatten()
    print(f"  ✓ Rede Neural concluída! ({len(history.epoch)} épocas executadas)")
    return modelo, history, predicoes


def treinar_todos_modelos(
    dados: DatasetModelagem,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Executa o treinamento de todas as arquiteturas (XGBoost, LightGBM, Rede Neural).

    Returns:
        Tupla com (dicionario_modelos, dicionario_predicoes, extras_como_historico)
    """
    print("\n" + "=" * 60)
    print(" ETAPA 5: MODELAGEM PREDITIVA (MACHINE LEARNING & DEEP LEARNING)")
    print("=" * 60)

    # 1. XGBoost
    modelo_xgb, pred_xgb = treinar_xgboost(
        dados.X_train_scaled, dados.y_train, dados.X_test_scaled, dados.y_test
    )

    # 2. LightGBM
    modelo_lgb, pred_lgb = treinar_lightgbm(
        dados.X_train_scaled, dados.y_train, dados.X_test_scaled, dados.y_test
    )

    # 3. Rede Neural MLP
    modelo_nn, history_nn, pred_nn = treinar_rede_neural(
        dados.X_train_scaled, dados.y_train, dados.X_test_scaled, dados.y_test
    )

    modelos = {
        "XGBoost": modelo_xgb,
        "LightGBM": modelo_lgb,
        "Rede Neural (MLP)": modelo_nn,
    }

    predicoes = {
        "XGBoost": pred_xgb,
        "LightGBM": pred_lgb,
        "Rede Neural (MLP)": pred_nn,
    }

    extras = {
        "history_nn": history_nn,
    }

    return modelos, predicoes, extras
