

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    ATIVOS,
    DATA_DIR,
    DATA_FIM,
    DATA_INICIO,
    INTERVALO,
    JANELA_MEDIA_CURTA,
    JANELA_MEDIA_LONGA,
    JANELA_RSI,
    JANELA_VOLATILIDADE,
    JANELA_VOLUME_MM,
)


def calcular_rsi(series: pd.Series, periodo: int = JANELA_RSI) -> pd.Series:
   
    delta = series.diff()
    ganho = delta.clip(lower=0).rolling(periodo).mean()
    perda = (-delta.clip(upper=0)).rolling(periodo).mean()
    rs = ganho / perda.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engenharia_atributos(df: pd.DataFrame, janela_volatilidade: int = JANELA_VOLATILIDADE) -> pd.DataFrame:
    """
    Adiciona indicadores financeiros derivados aos dados históricos:
    - Retorno Diário percentual
    - Volatilidade Histórica anualizada (sqrt(252))
    - Médias Móveis de 20 e 50 dias
    - RSI de 14 dias
    - Média móvel de volume (20 dias)

    Args:
        df: DataFrame com colunas ['Close', 'Volume', ...].
        janela_volatilidade: Período em dias úteis para volatilidade móvel.

    Returns:
        DataFrame enriquecido com os novos atributos técnicos.
    """
    df = df.copy()
    df["Retorno_Diario"] = df["Close"].pct_change()
    df[f"Volatilidade_{janela_volatilidade}d"] = (
        df["Retorno_Diario"].rolling(janela_volatilidade).std() * np.sqrt(252)
    )
    df[f"MM_{JANELA_MEDIA_CURTA}d"] = df["Close"].rolling(JANELA_MEDIA_CURTA).mean()
    df[f"MM_{JANELA_MEDIA_LONGA}d"] = df["Close"].rolling(JANELA_MEDIA_LONGA).mean()
    df[f"RSI_{JANELA_RSI}d"] = calcular_rsi(df["Close"], JANELA_RSI)
    df[f"Volume_MM{JANELA_VOLUME_MM}"] = df["Volume"].rolling(JANELA_VOLUME_MM).mean()
    return df


def baixar_ativo(
    ticker: str,
    inicio: str = DATA_INICIO,
    fim: str = DATA_FIM,
    intervalo: str = INTERVALO,
) -> Optional[pd.DataFrame]:
    """
    Baixa o histórico de preços ajustados para um ticker específico.

    Args:
        ticker: Símbolo do ativo (ex: '^BVSP', 'PETR4.SA').
        inicio: Data inicial 'YYYY-MM-DD'.
        fim: Data final 'YYYY-MM-DD'.
        intervalo: Intervalo de amostragem ('1d').

    Returns:
        DataFrame com colunas padronizadas ou None em caso de falha.
    """
    try:
        df = yf.download(
            ticker,
            start=inicio,
            end=fim,
            interval=intervalo,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            print(f"  [AVISO] Sem dados encontrados para: {ticker}")
            return None

        # Trata formato MultiIndex que o yfinance moderno pode retornar
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Data"
        df.reset_index(inplace=True)

        data_min = df["Data"].min().date() if hasattr(df["Data"].min(), "date") else df["Data"].min()
        data_max = df["Data"].max().date() if hasattr(df["Data"].max(), "date") else df["Data"].max()

        print(f"  {ticker:<15} {len(df)} registros ({data_min} -> {data_max})")
        return df

    except Exception as e:
        print(f"  [ERRO] Falha ao baixar {ticker}: {e}")
        return None


def salvar_ativo_csv(df: pd.DataFrame, ticker: str, categoria: str, pasta_saida: Path = DATA_DIR) -> Path:
    """
    Salva os dados do ativo em formato CSV na subpasta da categoria.
    """
    pasta_cat = pasta_saida / categoria
    pasta_cat.mkdir(parents=True, exist_ok=True)
    nome_arquivo = ticker.replace("^", "IDX_").replace(".SA", "") + ".csv"
    caminho = pasta_cat / nome_arquivo
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    return caminho


def coletar_todos_ativos(
    ativos: Dict[str, list] = ATIVOS,
    inicio: str = DATA_INICIO,
    fim: str = DATA_FIM,
    intervalo: str = INTERVALO,
    pasta_saida: Path = DATA_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Executa a rotina completa de coleta e cálculo de atributos para todos os ativos.

    Returns:
        Dicionário mapeando ticker -> DataFrame enriquecido.
    """
    dados: Dict[str, pd.DataFrame] = {}

    for categoria, tickers in ativos.items():
        print(f"\n--- Coletando Categoria: {categoria.upper()} ---")
        for ticker in tickers:
            df = baixar_ativo(ticker, inicio, fim, intervalo)
            if df is not None:
                df = engenharia_atributos(df)
                salvar_ativo_csv(df, ticker, categoria, pasta_saida)
                dados[ticker] = df

    print(f"\n[OK] Coleta concluída com sucesso! {len(dados)} ativos carregados e salvos em {pasta_saida}.")
    return dados


def carregar_ativo_local(ticker: str, categoria: Optional[str] = None, pasta_saida: Path = DATA_DIR) -> Optional[pd.DataFrame]:
    """
    Tenta carregar o DataFrame de um ativo a partir dos arquivos CSV salvos localmente.
    """
    nome_arquivo = ticker.replace("^", "IDX_").replace(".SA", "") + ".csv"

    if categoria:
        caminho = pasta_saida / categoria / nome_arquivo
        if caminho.exists():
            return pd.read_csv(caminho, parse_dates=["Data"])

    # Se não especificou categoria ou não encontrou, busca recursivamente
    arquivos_encontrados = list(pasta_saida.glob(f"**/{nome_arquivo}"))
    if arquivos_encontrados:
        return pd.read_csv(arquivos_encontrados[0], parse_dates=["Data"])

    return None
