"""
Módulo de Geração de Relatórios e Resumos Executivos.

Consolida os resultados numéricos do TCC, destacando o melhor modelo preditivo.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_FIM, DATA_INICIO, FIGURES_DIR, TICKER_REF


def gerar_relatorio_final(
    df_resultados: pd.DataFrame,
    ticker_ref: str = TICKER_REF,
    data_inicio: str = DATA_INICIO,
    data_fim: str = DATA_FIM,
    n_train: int = 0,
    n_test: int = 0,
    salvar: bool = True,
    pasta_saida: Path = FIGURES_DIR,
) -> str:
    """
    Formata e exibe o relatório final de desempenho dos modelos.
    """
    if df_resultados.empty:
        return ""

    melhor = df_resultados.iloc[0]
    linhas = [
        "=" * 64,
        "  RELATÓRIO FINAL — TCC Lorrana Evelyn de Araújo Pereira",
        "  Curso: Tecnologia em Gestão de Dados — UFPI / CEAD",
        f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "=" * 64,
        f"\n  Ativo analisado : {ticker_ref}",
        f"  Período         : {data_inicio} -> {data_fim}",
        f"  Amostras treino : {n_train}",
        f"  Amostras teste  : {n_test}",
        f"  Variável-alvo   : Volatilidade histórica (21 dias) — t+1",
        f"\n  {'Modelo':<22} {'RMSE':>10}  {'MAE':>10}  {'R²':>8}",
        "  " + "-" * 54,
    ]

    for _, row in df_resultados.iterrows():
        flag = " ⭐ (Melhor)" if row["Modelo"] == melhor["Modelo"] else ""
        linhas.append(
            f"  {row['Modelo']:<22} {row['RMSE']:>10.5f}  {row['MAE']:>10.5f}  {row['R2']:>8.4f}{flag}"
        )

    linhas.extend(
        [
            f"\n  Melhor modelo   : {melhor['Modelo']} (menor RMSE)",
            "=" * 64,
        ]
    )

    texto_relatorio = "\n".join(linhas)
    print("\n" + texto_relatorio)

    if salvar:
        caminho = pasta_saida / "relatorio_final_modelos.txt"
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(texto_relatorio)
        print(f"\n  [Relatório Salvo] {caminho}")

    return texto_relatorio
