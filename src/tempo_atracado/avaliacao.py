"""Tabelas de resultado, bootstrap pareado e erro desagregado."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rotulo(modelo, perfil):
    return modelo if perfil in ("—", "sem") else modelo + " + perfil"


def montar_tabela(resultados, mae_base):
    t = pd.DataFrame(resultados)
    t["Redução do MAE vs. referência (%)"] = 100 * (1 - t["MAE"] / mae_base)
    t = t.sort_values("MAE").reset_index(drop=True)
    t["Rótulo"] = [rotulo(m, p) for m, p in zip(t["Modelo"], t["Perfil K-Means"])]
    return t


def bootstrap_pareado(y, predicoes, rotulos, n_bootstrap, seed):
    """IC95% da diferença de MAE entre todos os pares, reamostrando as mesmas atracações."""
    E = np.column_stack([np.abs(y - predicoes[r]) for r in rotulos]).astype(np.float32)
    n_te, n_mod = E.shape
    rng = np.random.default_rng(seed)
    mae_boot = np.empty((n_bootstrap, n_mod))
    lote = max(1, int(2e7 // max(n_te * n_mod, 1)))       # limita a memória por lote
    for ini in range(0, n_bootstrap, lote):
        fim = min(n_bootstrap, ini + lote)
        idx = rng.integers(0, n_te, size=(fim - ini, n_te))
        mae_boot[ini:fim] = E[idx].mean(axis=1)

    linhas = []
    for i, a in enumerate(rotulos):
        for j, b in enumerate(rotulos):
            if i < j:
                d = mae_boot[:, j] - mae_boot[:, i]
                lo, hi = np.percentile(d, [2.5, 97.5])
                linhas.append({"Modelo A": a, "Modelo B": b,
                               "MAE(B) − MAE(A) (h)": float(E[:, j].mean() - E[:, i].mean()),
                               "IC95% inf": lo, "IC95% sup": hi, "Significativa": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(linhas)


def mae_por(av, col):
    t = (av.groupby(col).agg(n=("ta", "size"), TA_mediano_h=("ta", "median"), MAE=("erro_abs", "mean"),
                             vies_medio_h=("erro", "mean")).sort_values("n", ascending=False))
    t["MAE_relativo_pct"] = 100 * t["MAE"] / t["TA_mediano_h"]
    return t


def _rotulos_n(indice, n):
    return [f"{p} (n = {k:,})".replace(",", ".") for p, k in zip(indice, n)]


def fig_mae_porto_carga(por_porto, por_natureza, melhor, ano_teste):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    top = por_porto.head(15).iloc[::-1]
    ax[0].barh(_rotulos_n(top.index, top["n"]), top["MAE"], color="#3b6ea8")
    ax[0].set_xlabel("MAE (h)"); ax[0].set_title("Por instalação portuária (15 com mais atracações)")
    nat = por_natureza.iloc[::-1]
    ax[1].barh(_rotulos_n(nat.index, nat["n"]), nat["MAE"], color="#3b6ea8")
    ax[1].set_xlabel("MAE (h)"); ax[1].set_title("Por natureza da carga")
    fig.suptitle(f"MAE do melhor modelo ({melhor}) — teste {ano_teste}")
    fig.tight_layout()
    return fig


def fig_mes_e_residuos(por_mes, av, ano_teste):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(por_mes.index, por_mes["MAE"], "o-", color="#3b6ea8")
    ax[0].set_xlabel(f"Mês de {ano_teste}"); ax[0].set_ylabel("MAE (h)"); ax[0].set_title("MAE por mês do teste")
    ax[0].set_xticks(range(1, 13))
    faixas = pd.qcut(av["ta"], 10, duplicates="drop")
    res_f = av.groupby(faixas, observed=True)["erro"].agg(["mean", "median"])
    ax[1].plot(range(len(res_f)), res_f["mean"], "o-", label="média")
    ax[1].plot(range(len(res_f)), res_f["median"], "s--", label="mediana")
    ax[1].axhline(0, c="gray", lw=0.8)
    ax[1].set_xticks(range(len(res_f)))
    ax[1].set_xticklabels([f"{iv.right:.0f}" for iv in res_f.index], rotation=45)
    ax[1].set_xlabel("Decil do TA observado (limite superior, h)"); ax[1].set_ylabel("Erro (predito − observado, h)")
    ax[1].set_title("Resíduos por faixa de TA"); ax[1].legend()
    fig.tight_layout()
    return fig


def fig_distribuicao_ta(treino):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(treino["ta"], bins=80, color="#3b6ea8"); ax[0].set_xlabel("TA (h)"); ax[0].set_ylabel("Atracações")
    ax[0].set_title(f"TA — assimetria = {treino['ta'].skew():.2f}")
    ax[1].hist(np.log1p(treino["ta"]), bins=80, color="#3b6ea8"); ax[1].set_xlabel("log(1 + TA)")
    ax[1].set_title(f"log(1 + TA) — assimetria = {np.log1p(treino['ta']).skew():.2f}")
    fig.suptitle("Distribuição do tempo atracado no treino")
    fig.tight_layout()          # evita o título geral sobreposto aos títulos dos painéis
    return fig
