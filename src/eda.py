"""
Módulo de Análise Exploratória de Dados (EDA).

Gera visualizações estatísticas e financeiras:
- Fechamento com Médias Móveis (Plotly)
- Série histórica da Volatilidade Anualizada (Plotly)
- Distribuição de Retornos e Q-Q Plot (Matplotlib / Seaborn / Scipy)
- Matriz de Correlação de Retornos entre Ativos (Seaborn Heatmap)
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from src.config import FIGURES_DIR, TICKER_REF

# Estilo padrão dos gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.family"] = "DejaVu Sans"


def obter_estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna tabela com média, desvio padrão, quartis e extremos das variáveis numéricas.
    """
    colunas = [c for c in ["Close", "Volume", "Retorno_Diario", "Volatilidade_21d", "MM_20d", "MM_50d", "RSI_14d"] if c in df.columns]
    desc = df[colunas].describe().round(4)
    return desc


def plotar_fechamento_medias(
    df: pd.DataFrame,
    ticker: str = TICKER_REF,
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> go.Figure:
    """
    Gráfico interativo com Preço de Fechamento e Médias Móveis (20d e 50d).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Data"],
            y=df["Close"],
            name="Fechamento",
            line=dict(color="#1f77b4", width=1.2),
        )
    )
    if "MM_20d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Data"],
                y=df["MM_20d"],
                name="MM 20d",
                line=dict(color="#ff7f0e", dash="dash"),
            )
        )
    if "MM_50d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Data"],
                y=df["MM_50d"],
                name="MM 50d",
                line=dict(color="#2ca02c", dash="dot"),
            )
        )

    fig.update_layout(
        title=f"{ticker} — Preço de Fechamento e Médias Móveis",
        xaxis_title="Data",
        yaxis_title="Preço / Pontos",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", y=1.02),
    )

    if salvar:
        caminho = pasta_figuras / f"eda_fechamento_medias_{ticker.replace('^', '').replace('.SA', '')}.html"
        fig.write_html(str(caminho))
        print(f"  [Gráfico Salvo] {caminho}")

    if mostrar:
        fig.show()

    return fig


def plotar_volatilidade_historica(
    df: pd.DataFrame,
    ticker: str = TICKER_REF,
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> go.Figure:
    """
    Gráfico interativo em área destacando a evolução da volatilidade anualizada (janela 21 dias).
    """
    col_vol = [c for c in df.columns if c.startswith("Volatilidade_")][0]

    fig = px.area(
        df,
        x="Data",
        y=col_vol,
        title=f"Volatilidade Histórica Anualizada — {ticker} ({col_vol.replace('_', ' ')})",
        labels={col_vol: "Volatilidade (anualizada)"},
        color_discrete_sequence=["#d62728"],
        template="plotly_white",
        height=400,
    )
    fig.update_traces(fillcolor="rgba(214,39,40,0.15)")

    if salvar:
        caminho = pasta_figuras / f"eda_volatilidade_{ticker.replace('^', '').replace('.SA', '')}.html"
        fig.write_html(str(caminho))
        print(f"  [Gráfico Salvo] {caminho}")

    if mostrar:
        fig.show()

    return fig


def plotar_distribuicao_retornos(
    df: pd.DataFrame,
    ticker: str = TICKER_REF,
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> plt.Figure:
    """
    Gera histograma com KDE e Q-Q Plot comparando com a distribuição Normal.
    Calcula e exibe assimetria (skewness) e curtose (kurtosis).
    """
    retornos = df["Retorno_Diario"].dropna()
    skew = retornos.skew()
    kurt = retornos.kurtosis()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histograma + KDE
    sns.histplot(retornos, bins=80, kde=True, color="steelblue", ax=axes[0])
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"Distribuição dos Retornos Diários — {ticker}")
    axes[0].set_xlabel("Retorno Diário")
    axes[0].set_ylabel("Frequência")

    # Q-Q Plot
    stats.probplot(retornos, plot=axes[1])
    axes[1].set_title("Q-Q Plot — Retornos vs. Distribuição Normal")

    plt.tight_layout()

    print(f"\n  [Estatísticas de Forma - {ticker}]")
    print(f"  Assimetria (skewness): {skew:.4f}")
    print(f"  Curtose (kurtosis):    {kurt:.4f}")
    if kurt > 3:
        print("  -> Curtose elevada confirma caudas pesadas (leptocúrtica) — típico de séries financeiras.")

    if salvar:
        caminho = pasta_figuras / f"eda_distribuicao_retornos_{ticker.replace('^', '').replace('.SA', '')}.png"
        fig.savefig(caminho, bbox_inches="tight")
        print(f"  [Gráfico Salvo] {caminho}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plotar_matriz_correlacao(
    dados: Dict[str, pd.DataFrame],
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> Optional[plt.Figure]:
    """
    Gera heatmap com a correlação dos retornos diários entre todos os ativos carregados.
    """
    retornos_dict = {}
    for ticker, df in dados.items():
        if "Retorno_Diario" in df.columns:
            nome = ticker.replace(".SA", "").replace("^", "")
            retornos_dict[nome] = df.set_index("Data")["Retorno_Diario"]

    if not retornos_dict:
        print("  [AVISO] Sem retornos disponíveis para gerar matriz de correlação.")
        return None

    df_retornos = pd.DataFrame(retornos_dict).dropna()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    corr = df_retornos.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlação de Retornos entre Ativos", fontsize=13, pad=15)
    plt.tight_layout()

    if salvar:
        caminho = pasta_figuras / "eda_matriz_correlacao_ativos.png"
        fig.savefig(caminho, bbox_inches="tight")
        print(f"  [Gráfico Salvo] {caminho}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return fig


def executar_eda_completa(
    df_ref: pd.DataFrame,
    dados_todos: Optional[Dict[str, pd.DataFrame]] = None,
    ticker_ref: str = TICKER_REF,
    salvar: bool = True,
    mostrar: bool = True,
) -> None:
    """
    Pipeline que executa todos os passos de EDA sequencialmente.
    """
    print("\n" + "=" * 60)
    print(f" ETAPA 3: ANÁLISE EXPLORATÓRIA DOS DADOS (EDA) — {ticker_ref}")
    print("=" * 60)

    print(f"\n[Shape]: {df_ref.shape}")
    dt_min = df_ref["Data"].min().date() if hasattr(df_ref["Data"].min(), "date") else df_ref["Data"].min()
    dt_max = df_ref["Data"].max().date() if hasattr(df_ref["Data"].max(), "date") else df_ref["Data"].max()
    print(f"[Período]: {dt_min} -> {dt_max}")

    print("\n[Estatísticas Descritivas]:")
    print(obter_estatisticas_descritivas(df_ref))

    print("\n-> Gerando gráficos de fechamento e médias...")
    plotar_fechamento_medias(df_ref, ticker_ref, salvar=salvar, mostrar=mostrar)

    print("-> Gerando gráfico de volatilidade histórica...")
    plotar_volatilidade_historica(df_ref, ticker_ref, salvar=salvar, mostrar=mostrar)

    print("-> Gerando distribuição dos retornos e Q-Q Plot...")
    plotar_distribuicao_retornos(df_ref, ticker_ref, salvar=salvar, mostrar=mostrar)

    if dados_todos:
        print("-> Gerando matriz de correlação de ativos...")
        plotar_matriz_correlacao(dados_todos, salvar=salvar, mostrar=mostrar)
