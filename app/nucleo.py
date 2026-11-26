"""Carregamento do modelo exportado pelo pipeline e cálculo das predições.

O arquivo `resultados/modelo_streamlit.joblib` é gerado na etapa 12 do pipeline
(`python -m tempo_atracado`) e contém o melhor modelo do experimento, o K-Means do
perfil operacional, as tabelas da referência ingênua e os metadados de avaliação.
"""
from __future__ import annotations

import os
import unicodedata

import joblib
import numpy as np
import pandas as pd

# Colunas que o usuário informa na interface (antes das transformações)
ENTRADAS = ["porto", "berco", "tipo_navegacao", "natureza", "armador", "mes",
            "peso_total", "teu_total", "n_registros_carga"]

MESES = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
         7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"}

NIVEIS_REF = [("berço × tipo de carga", ["berco_id", "natureza"]), ("berço", ["berco_id"]), ("porto", ["porto"])]


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e sem separadores — mesma regra usada na preparação dos dados."""
    s = unicodedata.normalize("NFKD", str(texto).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


# ----------------------------------------------------------------- carregamento
def caminho_padrao() -> str:
    """`resultados/modelo_streamlit.joblib` na raiz do repositório."""
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("MODELO_TA", os.path.join(raiz, "resultados", "modelo_streamlit.joblib"))


def dir_resultados() -> str:
    return os.path.dirname(caminho_padrao())


def carregar_bundle(caminho: str | None = None) -> dict:
    """Lê o modelo exportado e deixa as tabelas da referência prontas para consulta."""
    bundle = joblib.load(caminho or caminho_padrao())
    bundle["bercos_df"] = pd.DataFrame(bundle["bercos"]).sort_values(["porto", "berco_id"])
    bundle["ref_series"] = {nome: _serie_referencia(tab) for nome, tab in bundle["ref_tabelas"].items()}
    return bundle


def _serie_referencia(tabela: dict) -> pd.Series:
    """Dicionário salvo no bundle → Series com MultiIndex quando a chave é composta."""
    s = pd.Series(tabela, dtype=float)
    if len(s) and isinstance(s.index[0], tuple):
        s.index = pd.MultiIndex.from_tuples(s.index)
    return s


def portos(bundle: dict) -> list[str]:
    return sorted(bundle["bercos_df"]["porto"].unique().tolist())


def bercos_do_porto(bundle: dict, porto: str) -> pd.DataFrame:
    """Berços com histórico no treino, do mais movimentado para o menos."""
    d = bundle["bercos_df"]
    return d[d["porto"] == porto].sort_values("n", ascending=False)


def rotulo_berco(berco_id: str) -> str:
    """'Porto | Berço' → 'Berço'."""
    return berco_id.split(" | ", 1)[1] if " | " in berco_id else berco_id


# ----------------------------------------------------------------- atributos
def montar_atributos(bundle: dict, entradas: pd.DataFrame) -> pd.DataFrame:
    """Reproduz a engenharia de atributos do pipeline a partir dos campos da interface."""
    d = entradas.copy()
    composto = d["porto"].astype(str) + " | " + d["berco"].astype(str) if "berco" in d else None
    if "berco_id" not in d:
        d["berco_id"] = composto
    elif composto is not None:
        d["berco_id"] = d["berco_id"].fillna(composto)
    d["mes"] = d["mes"].astype(int)
    d["log_peso"] = np.log1p(d["peso_total"].clip(lower=0))
    d["log_teu"] = np.log1p(d["teu_total"].clip(lower=0))
    d["log_registros_carga"] = np.log1p(d["n_registros_carga"].clip(lower=0))

    # Frações de peso por natureza: a interface informa só a natureza predominante,
    # então a fração dela vale 1 e as demais 0 (simplificação registrada na seção "Sobre").
    for col in bundle["frac_cols"]:
        d[col] = 0.0
    mapa = {normalizar(nat): col for col in bundle["frac_cols"]
            for nat in [col[len("frac_"):]]}
    for i, nat in d["natureza"].items():
        col = mapa.get(normalizar(nat))
        if col:
            d.loc[i, col] = 1.0

    if bundle["usa_perfil"] and bundle.get("kmeans") is not None:
        Z = bundle["scaler_perfil"].transform(d[bundle["perfil_cols"]])
        d["perfil"] = "P" + bundle["kmeans"].predict(Z).astype(str)
    return d


def colunas_modelo(bundle: dict) -> list[str]:
    cats = list(bundle["cat_cols"]) + (["perfil"] if bundle["usa_perfil"] else [])
    return cats + list(bundle["time_cols"]) + list(bundle["num_cols"])


def prever(bundle: dict, entradas: pd.DataFrame) -> pd.DataFrame:
    """Devolve as entradas com o TA previsto (h), a referência ingênua e o nível da cascata usado."""
    d = montar_atributos(bundle, entradas)
    d["ta_previsto"] = np.clip(np.expm1(bundle["modelo"].predict(d[colunas_modelo(bundle)])), 0, None)
    ref, nivel = referencia_ingenua(bundle, d)
    d["ta_referencia"] = ref
    d["nivel_referencia"] = nivel
    d["mae_instalacao"] = d["porto"].map(bundle["mae_por_porto"]).fillna(bundle["metricas_teste"]["MAE"])
    d["berco_sem_historico"] = ~d["berco_id"].isin(set(bundle["bercos_df"]["berco_id"]))
    return d


def referencia_ingenua(bundle: dict, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mediana histórica em cascata: berço × carga → berço → porto → geral."""
    pred = np.full(len(d), np.nan)
    nivel = np.full(len(d), "geral", dtype=object)
    for nome, cols in NIVEIS_REF:
        tabela = bundle["ref_series"].get(nome)
        if tabela is None:
            continue
        falta = np.isnan(pred)
        if not falta.any():
            break
        alvo = d.loc[falta, cols]
        chaves = pd.MultiIndex.from_frame(alvo) if len(cols) > 1 else pd.Index(alvo[cols[0]])
        valores = tabela.reindex(chaves).values.astype(float)
        idx = np.where(falta)[0]
        achou = ~np.isnan(valores)
        pred[idx[achou]] = valores[achou]
        nivel[idx[achou]] = nome
    pred[np.isnan(pred)] = bundle["ref_global"]
    return pred, nivel


# ----------------------------------------------------------------- utilidades
def formatar_horas(h: float) -> str:
    """36.5 → '36h30 (1,5 dia)'."""
    horas, minutos = int(h), int(round((h - int(h)) * 60))
    if minutos == 60:
        horas, minutos = horas + 1, 0
    dias = round(h / 24, 1)
    plural = "" if dias == 1.0 else "s"
    return f"{horas}h{minutos:02d} ({f'{dias:.1f}'.replace('.', ',')} dia{plural})"


def tabela_calibracao(predicoes: pd.DataFrame, coluna: str, n_faixas: int = 10) -> pd.DataFrame:
    """Divide as predições do teste em faixas (decis) e resume o TA observado em cada uma.

    Serve para mostrar, para uma nova estimativa, o que de fato aconteceu no teste com as
    atracações que receberam previsões parecidas. Atenção: no teste o volume era o realizado.
    """
    d = predicoes[[coluna, "ta"]].dropna()
    faixas = pd.qcut(d[coluna], n_faixas, duplicates="drop")
    g = d.groupby(faixas, observed=True)
    t = pd.DataFrame({
        "pred_min": g[coluna].min(), "pred_max": g[coluna].max(), "n": g.size(),
        "obs_p10": g["ta"].quantile(0.10), "obs_mediana": g["ta"].median(), "obs_p90": g["ta"].quantile(0.90),
        "pct_dentro_12h": g.apply(lambda x: 100 * ((x["ta"] - x[coluna]).abs() <= 12).mean()),
        "pct_acima_120h": g.apply(lambda x: 100 * (x["ta"] > 120).mean()),
    }).reset_index(drop=True)
    return t


def intervalo_empirico(calibracao: pd.DataFrame, previsto: float) -> dict:
    """Linha da tabela de calibração cuja faixa de predições contém o valor previsto."""
    i = int(np.searchsorted(calibracao["pred_max"].values, previsto, side="left"))
    return calibracao.iloc[min(i, len(calibracao) - 1)].to_dict()


def ler_csv_resultado(nome: str) -> pd.DataFrame | None:
    """Lê uma tabela da pasta `resultados/`, se existir."""
    caminho = os.path.join(dir_resultados(), nome)
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho)
