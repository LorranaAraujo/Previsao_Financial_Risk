"""
Módulo de Pré-processamento e Preparação dos Dados.

Realiza:
- Construção da variável-alvo (Volatilidade do dia seguinte: t+1)
- Divisão temporal em treino e teste (sem shuffle para evitar vazamento de dados)
- Padronização das features (StandardScaler ajustado apenas no conjunto de treino)
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, TARGET, TEST_SIZE


@dataclass
class DatasetModelagem:
    """Estrutura tipada para transportar os dados preparados para treino e teste."""
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    datas_train: np.ndarray
    datas_test: np.ndarray
    features: List[str]
    target: str
    scaler: StandardScaler
    df_model: pd.DataFrame

    @property
    def X(self) -> np.ndarray:
        """Matriz completa de features sem escala (X_train seguido de X_test)."""
        return np.vstack([self.X_train, self.X_test])

    @property
    def y(self) -> np.ndarray:
        """Vetor completo da variável-alvo (y_train seguido de y_test)."""
        return np.concatenate([self.y_train, self.y_test])

    @property
    def datas(self) -> np.ndarray:
        """Array completo de datas sequenciais."""
        return np.concatenate([self.datas_train, self.datas_test])


def preparar_dados_modelagem(
    df: pd.DataFrame,
    features: List[str] = FEATURES,
    coluna_volatilidade: str = "Volatilidade_21d",
    target_name: str = TARGET,
    test_size: float = TEST_SIZE,
) -> DatasetModelagem:
    """
    Prepara o dataset para os algoritmos de regressão supervisionada.

    Passos:
    1. Criação do Target com shift(-1) para prever a volatilidade do dia seguinte.
    2. Remoção de valores nulos (gerados por rolling windows e shift).
    3. Divisão temporal em Treino e Teste (80% / 20%).
    4. Normalização com StandardScaler treinado exclusivamente no split de treino.

    Args:
        df: DataFrame com indicadores técnicos e coluna 'Data'.
        features: Lista com os nomes dos preditores numéricos.
        coluna_volatilidade: Nome da coluna base para o target.
        target_name: Nome da coluna-alvo resultante.
        test_size: Proporção para conjunto de teste (ex: 0.20 para 80/20).

    Returns:
        Instância de DatasetModelagem contendo arrays e scaler.
    """
    df_model = df[features + ["Data"]].copy()

    # Previsão 1 passo à frente: Target = volatilidade no período seguinte (t+1)
    df_model[target_name] = df[coluna_volatilidade].shift(-1)
    df_model.dropna(inplace=True)

    X = df_model[features].values
    y = df_model[target_name].values
    datas = df_model["Data"].values

    # Split temporal sequencial
    split_idx = int(len(X) * (1.0 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    datas_train = datas[:split_idx]
    datas_test = datas[split_idx:]

    # Padronização (ajuste apenas no treino para evitar data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print(" ETAPA 4: PRÉ-PROCESSAMENTO & PREPARAÇÃO PARA MODELAGEM")
    print("=" * 60)
    print(f"  Amostras Treino : {X_train.shape[0]} ({100*(1-test_size):.0f}%)")
    print(f"  Amostras Teste  : {X_test.shape[0]} ({100*test_size:.0f}%)")
    print(f"  Features ({len(features)}) : {features}")
    print(f"  Target         : {target_name} (Volatilidade do dia seguinte - t+1)")
    print("=" * 60)

    return DatasetModelagem(
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        datas_train=datas_train,
        datas_test=datas_test,
        features=features,
        target=target_name,
        scaler=scaler,
        df_model=df_model,
    )
