"""Tabelas de resultado, bootstrap pareado, erro desagregado e utilidade operacional."""
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


def bootstrap_dif(erro_a, erro_b, n_bootstrap, seed):
    """Diferença MAE(b) − MAE(a) e IC95%, reamostrando as mesmas atracações do teste."""
    ea, eb = np.asarray(erro_a, dtype=np.float32), np.asarray(erro_b, dtype=np.float32)
    rng = np.random.default_rng(seed)
    difs = np.empty(n_bootstrap)
    lote = max(1, int(2e7 // max(len(ea), 1)))            # limita a memória por lote
    for ini in range(0, n_bootstrap, lote):
        fim = min(n_bootstrap, ini + lote)
        idx = rng.integers(0, len(ea), size=(fim - ini, len(ea)))
        difs[ini:fim] = eb[idx].mean(1) - ea[idx].mean(1)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return float(eb.mean() - ea.mean()), float(lo), float(hi)


# ------------------------------------------------------------------ Utilidade operacional
FAIXAS_TA = [0, 12, 24, 48, 120, np.inf]
ROTULOS_FAIXAS = ["até 12 h", "12–24 h", "24–48 h", "48–120 h", "acima de 120 h"]


def utilidade(y, p):
    """Proporção de predições dentro de tolerâncias fixas, além do erro médio e mediano."""
    e = np.abs(y - p)
    return {"MAE": float(e.mean()), "MedAE (h)": float(np.median(e)),
            "% dentro de ±6 h": 100 * float((e <= 6).mean()),
            "% dentro de ±12 h": 100 * float((e <= 12).mean()),
            "% dentro de ±24 h": 100 * float((e <= 24).mean()),
            "Erro relativo mediano (%)": 100 * float(np.median(e / y)),
            "% observado acima do previsto": 100 * float((y > p).mean())}


def erro_por_faixa(y, pred, pred_ref):
    """Erro e viés do modelo por faixa de duração observada, com a referência ao lado."""
    e, e_ref = np.abs(y - pred), np.abs(y - pred_ref)
    g = pd.DataFrame({"faixa": pd.cut(y, FAIXAS_TA, labels=ROTULOS_FAIXAS, right=True),
                      "e": e, "vies": pred - y, "e_ref": e_ref})
    t = g.groupby("faixa", observed=True).agg(
        n=("e", "size"), MAE=("e", "mean"), vies_medio_h=("vies", "mean"), MAE_referencia=("e_ref", "mean"),
        pct_dentro_12h=("e", lambda s: 100 * (s <= 12).mean()),
        pct_dentro_24h=("e", lambda s: 100 * (s <= 24).mean()))
    t["pct_do_teste"] = 100 * t["n"] / len(y)
    t["pct_da_soma_dos_erros"] = 100 * g.groupby("faixa", observed=True)["e"].sum() / e.sum()
    return t


def resumo_erro(y, p):
    """Métricas usadas para comparar configurações de corte de extremos."""
    from .modelos import metricas          # import tardio: evita ciclo de importação

    e = np.abs(y - p)
    longo = y > 120
    return {**metricas(y, p),
            "MedAE (h)": float(np.median(e)),
            "% dentro de ±12 h": 100 * float((e <= 12).mean()),
            "% dentro de ±24 h": 100 * float((e <= 24).mean()),
            "MAE TA ≤ 48 h": float(e[y <= 48].mean()),
            "MAE TA > 120 h": float(e[longo].mean()) if longo.any() else float("nan"),
            "Viés TA > 120 h": float((p - y)[longo].mean()) if longo.any() else float("nan"),
            "Maior previsão (h)": float(p.max())}


def diagnostico_mediana(y, z_pred):
    """Resíduos na escala logarítmica, que sustentam (ou não) a leitura da predição como mediana.

    `z_pred` é a predição na escala log(1 + TA), antes da reconversão para horas. Usar o valor já
    reconvertido reintroduziria o truncamento em zero de `para_horas` e distorceria o resíduo.
    """
    from scipy.stats import skew

    lr = np.log1p(y) - np.asarray(z_pred, dtype=float)
    acima = pd.Series(lr > 0)
    por_decil = acima.groupby(pd.qcut(np.asarray(z_pred, dtype=float), 10, duplicates="drop"),
                              observed=True).mean()
    return {"mediana do resíduo (log)": float(np.median(lr)),
            "média do resíduo (log)": float(np.mean(lr)),
            "assimetria do resíduo (log)": float(skew(lr)),
            "% observado acima do previsto": 100 * float(acima.mean()),
            "% acima do previsto, mín. entre decis da predição": 100 * float(por_decil.min()),
            "% acima do previsto, máx. entre decis da predição": 100 * float(por_decil.max())}


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
