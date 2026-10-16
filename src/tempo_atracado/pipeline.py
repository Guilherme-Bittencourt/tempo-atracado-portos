"""Orquestra o experimento completo, da leitura dos dados ao resumo para o artigo."""
import os
import platform
import traceback
import warnings
from dataclasses import asdict
from importlib import metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import avaliacao as av_
from . import dados
from . import modelos as mdl
from .config import CAT_COLS, NUM_COLS, TIME_COLS, Config
from .perfil import ajustar_perfis
from .referencia import ReferenciaIngenua
from .utils import Cronometro, Saida, mostrar


def executar(cfg: Config | None = None) -> dict:
    """Roda todas as etapas e devolve o resumo com os valores para o artigo."""
    cfg = cfg or Config()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    crono = Cronometro()
    saida = Saida(cfg.dir_saida, cfg.mostrar_figuras)
    os.makedirs(cfg.dir_modelos, exist_ok=True)
    np.random.seed(cfg.seed)
    saida.json({"config": asdict(cfg), "versoes": _versoes()}, "execucao.json")   # para reprodutibilidade
    print(f"Dados em: {os.path.abspath(cfg.dir_dados)}\nResultados em: {os.path.abspath(cfg.dir_saida)}")

    # ------------------------------------------------------------------ 1. Arquivos
    crono.etapa("1. Localizando arquivos")
    dados.extrair_zips(cfg)
    arquivos, usa_tempos = dados.localizar_arquivos(cfg)

    # ------------------------------------------------------------------ 2. Leitura
    crono.etapa("2. Leitura das bases")
    atr, tempos, carga, armador_fonte = dados.carregar_bases(cfg, arquivos, usa_tempos, saida)

    # ------------------------------------------------------------------ 3. Integração
    crono.etapa("3. Integração e atributos")
    df, contagem = dados.montar_base(atr, tempos, carga)
    del atr, tempos, carga
    frac_cols = sorted(c for c in df.columns if c.startswith("frac_"))
    perfil_cols = list(NUM_COLS) + frac_cols

    # ------------------------------------------------------------------ 4. Filtros e divisão
    crono.etapa("4. Filtros e divisão temporal")
    treino, teste, ta_max_h, mask_trunc = dados.filtrar_e_dividir(df, contagem, cfg)
    del df
    funil = dados.tabela_funil(contagem)
    saida.csv(funil, "funil_preparacao.csv")
    print("FUNIL DE PREPARAÇÃO:")
    mostrar(funil)
    if ta_max_h:
        print(f"Corte de extremos: TA > {ta_max_h:.1f} h (quantil {cfg.ta_quantil_max:.1%} do treino), "
              f"aplicado {'a treino e teste' if cfg.corte_no_teste else 'SÓ ao treino'}. "
              f"No teste completo há {int((~mask_trunc).sum()):,} atracações acima do corte.")

    base_final = pd.concat([treino, teste], ignore_index=True)
    descr = pd.DataFrame({"treino": treino["ta"].describe(), f"teste {cfg.ano_teste}": teste["ta"].describe(),
                          "conjunto final": base_final["ta"].describe()}).round(2)
    saida.csv(descr, "descritivas_ta.csv")
    print("\nDESCRITIVAS DO TA (h):")
    mostrar(descr)
    ta_med_nat = base_final.groupby("natureza")["ta"].median().round(1)
    print("TA mediano por natureza (conjunto final):", ta_med_nat.to_dict())
    n_instal, n_bercos = base_final["porto"].nunique(), base_final["berco_id"].nunique()
    print(f"Instalações distintas: {n_instal} | Berços distintos: {n_bercos}")
    if "tipo_autoridade" in base_final:
        print("Atracações por tipo de autoridade portuária (porto organizado x TUP):")
        print(base_final["tipo_autoridade"].value_counts().to_string())
    por_ano_carga = pd.crosstab(base_final["ano"], base_final["natureza"], margins=True)
    saida.csv(por_ano_carga, "atracacoes_por_ano_e_carga.csv")
    mostrar(por_ano_carga)

    teste["berco_novo"] = ~teste["berco_id"].isin(set(treino["berco_id"]))
    print(f"Atracações de {cfg.ano_teste} em berços sem histórico no treino: {int(teste['berco_novo'].sum()):,} "
          f"({100 * teste['berco_novo'].mean():.2f}%) em "
          f"{teste.loc[teste['berco_novo'], 'berco_id'].nunique()} berços")

    conf_ta = None
    if usa_tempos and "ta_datas" in base_final:
        dif = (base_final["ta"] - base_final["ta_datas"]).abs()
        conf_ta = {"comparaveis": int(dif.notna().sum()), "divergencia_maior_que_0_1h": int((dif > 0.1).sum()),
                   "pct": float(100 * (dif > 0.1).mean()), "dif_mediana_h": float(dif.median())}
        print(f"Conferência do TA (TAtracado x diferença de datas): {conf_ta}")
    saida.fig(av_.fig_distribuicao_ta(treino), "fig_distribuicao_ta.png")

    # ------------------------------------------------------------------ 5. Referência ingênua
    crono.etapa("5. Referência ingênua")
    ref = ReferenciaIngenua().fit(treino)
    pred_ref, nivel_ref = ref.predict(teste)
    cascata = (pd.Series(nivel_ref).value_counts().rename("atracações").to_frame()
               .assign(percentual=lambda t: (100 * t["atracações"] / len(teste)).round(2)))
    saida.csv(cascata, "cascata_referencia.csv")
    print("Nível da cascata usado no teste:")
    mostrar(cascata)

    # ------------------------------------------------------------------ 6. K-Means
    crono.etapa("6. Perfil operacional (K-Means)")
    perf = ajustar_perfis(treino, teste, perfil_cols, cfg, saida)
    mostrar(perf["caracterizacao"].round(2))
    k_usado = perf["k_usado"]

    # ------------------------------------------------------------------ 7. Modelos
    crono.etapa("7. Modelos (reaproveita os já treinados)")
    treino_tun = treino
    if cfg.amostra_tuning and len(treino) > cfg.amostra_tuning:
        idx_tun = np.sort(np.random.default_rng(cfg.seed).choice(len(treino), cfg.amostra_tuning, replace=False))
        treino_tun = treino.iloc[idx_tun]

    y_te = teste["ta"].values
    predicoes = {"Referência ingênua": pred_ref}
    melhores_params, salvos = {}, {}
    res_ref = {"Modelo": "Referência ingênua", "Perfil K-Means": "—", **mdl.metricas(y_te, pred_ref)}
    resultados = [res_ref]
    ass_padrao = mdl.assinatura(cfg, k_usado, len(treino))

    for usa_perfil in (False, True):
        for nome in cfg.modelos:
            rotulo = nome + (" + perfil" if usa_perfil else "")
            arq = mdl.arquivo_modelo(cfg.dir_modelos, rotulo)
            salvo = mdl.carregar_se_valido(arq, ass_padrao)
            if salvo is not None:
                print(f"[{rotulo}] carregado", flush=True)
            else:
                print(f"[{rotulo}] treinando...", flush=True)
                salvo = mdl.treinar_com_busca(nome, usa_perfil, treino_tun, treino, cfg, ass_padrao)
                joblib.dump(salvo, arq)
                print(f"[{rotulo}] busca {salvo['t_busca']:.0f}s | ajuste {salvo['t_fit']:.0f}s | "
                      f"MAE CV {salvo['cv_mae']:.2f} h   [{crono.minutos():.1f} min]", flush=True)
            salvos[rotulo] = salvo
            p = mdl.para_horas(salvo["modelo"].predict(mdl.matriz(teste, usa_perfil)))
            predicoes[rotulo] = p
            melhores_params[rotulo] = {k: (v.item() if hasattr(v, "item") else v) for k, v in salvo["params"].items()}
            resultados.append({"Modelo": nome, "Perfil K-Means": "com" if usa_perfil else "sem",
                               **mdl.metricas(y_te, p), "MAE CV (h)": salvo["cv_mae"]})
    saida.json(melhores_params, "melhores_hiperparametros.json")

    # ------------------------------------------------------------------ 8. Resultados no teste
    crono.etapa("8. Resultados no teste")
    mae_ref = res_ref["MAE"]
    tab = av_.montar_tabela(resultados, mae_ref)
    arred = {"MAE": 2, "RMSE": 2, "R2": 3, "MAE CV (h)": 2, "Redução do MAE vs. referência (%)": 1}
    saida.csv(tab, "tabela_resultados_teste_completo.csv", index=False)
    print(f"TESTE COMPLETO ({len(teste):,} atracações):")
    mostrar(tab.drop(columns="Rótulo").round(arred))

    res_trunc = [{"Modelo": r["Modelo"], "Perfil K-Means": r["Perfil K-Means"],
                  **mdl.metricas(y_te[mask_trunc], predicoes[av_.rotulo(r["Modelo"], r["Perfil K-Means"])][mask_trunc])}
                 for r in resultados]
    tab_trunc = av_.montar_tabela(res_trunc, res_trunc[0]["MAE"])
    saida.csv(tab_trunc, "tabela_resultados_teste_truncado.csv", index=False)
    print(f"Teste TRUNCADO no corte do treino ({int(mask_trunc.sum()):,} atracações), "
          f"comparável à versão original do experimento:")
    mostrar(tab_trunc.drop(columns="Rótulo").round(arred))

    ef = tab[tab["Perfil K-Means"] != "—"].pivot(index="Modelo", columns="Perfil K-Means", values=["MAE", "RMSE", "R2"])
    efeito = pd.DataFrame({f"Δ{m} (com − sem)": ef[(m, "com")] - ef[(m, "sem")] for m in ["MAE", "RMSE", "R2"]})
    saida.csv(efeito, "efeito_perfil_kmeans.csv")
    mostrar(efeito.round(4))

    melhor = [r for r in tab["Rótulo"] if r != "Referência ingênua"][0]
    nome_melhor = tab.loc[tab["Rótulo"] == melhor, "Modelo"].iat[0]
    usa_perfil_melhor = melhor.endswith("+ perfil")
    modelo_melhor = salvos[melhor]["modelo"]
    params_melhor = salvos[melhor]["params"]
    print(f">>> Melhor modelo no teste completo: {melhor} <<<")

    # ------------------------------------------------------------------ 8b. Ablação
    ablacao = []
    if cfg.rodar_ablacao:
        crono.etapa("8b. Ablação: melhor modelo sem os atributos de volume movimentado")
        variantes = {"sem peso e TEU": ("log_peso", "log_teu"),
                     "sem peso, TEU e nº de registros": ("log_peso", "log_teu", "log_registros_carga")}
        ablacao.append({"Variante": "completo", **mdl.metricas(y_te, predicoes[melhor])})
        for nome_var, sem in variantes.items():
            try:
                rot = f"ablacao {nome_melhor} {nome_var}" + (" perfil" if usa_perfil_melhor else "")
                arq = mdl.arquivo_modelo(cfg.dir_modelos, rot)
                ass = mdl.assinatura(cfg, k_usado, len(treino), extra=rot)
                salvo = mdl.carregar_se_valido(arq, ass)
                if salvo is None:
                    print(f"[{nome_var}] treinando (mesmos hiperparâmetros do modelo completo)...", flush=True)
                    modelo, _ = mdl.criar_modelo(nome_melhor, usa_perfil_melhor, sem=sem, seed=cfg.seed)
                    modelo.set_params(**params_melhor)
                    modelo.fit(mdl.matriz(treino, usa_perfil_melhor, sem), mdl.para_log(treino["ta"].values),
                               **mdl.fit_kwargs(nome_melhor, usa_perfil_melhor))
                    salvo = {"modelo": modelo, "params": params_melhor, "cv_mae": np.nan, "t_busca": 0,
                             "t_fit": 0, "assinatura": ass}
                    joblib.dump(salvo, arq)
                p = mdl.para_horas(salvo["modelo"].predict(mdl.matriz(teste, usa_perfil_melhor, sem)))
                predicoes[f"{nome_melhor} ({nome_var})"] = p
                ablacao.append({"Variante": nome_var, **mdl.metricas(y_te, p)})
            except Exception:
                print(f"Ablação '{nome_var}' falhou:")
                traceback.print_exc()
        ablacao.append({"Variante": "referência ingênua", **mdl.metricas(y_te, pred_ref)})
        tab_abl = pd.DataFrame(ablacao)
        tab_abl["Redução do MAE vs. referência (%)"] = 100 * (1 - tab_abl["MAE"] / mae_ref)
        saida.csv(tab_abl, "ablacao_volume.csv", index=False)
        mostrar(tab_abl.round({"MAE": 2, "RMSE": 2, "R2": 3, "Redução do MAE vs. referência (%)": 1}))
        for _, r in tab_abl.iloc[1:-1].iterrows():
            if r["MAE"] < mae_ref:
                print(f"  → '{r['Variante']}': ainda supera a referência em {100 * (1 - r['MAE'] / mae_ref):.1f}%. "
                      f"O modelo não depende só do volume.")
            else:
                print(f"  → '{r['Variante']}': NÃO supera a referência. O ganho depende da estimativa de volume.")

    # ------------------------------------------------------------------ 8c/8d. Opcionais
    extras = []
    if cfg.rodar_xgb_nativo:
        crono.etapa("8c. XGBoost com categóricas nativas (opcional)")
        try:
            extras.append(_xgb_nativo(cfg, treino, teste, y_te, predicoes, k_usado))
        except Exception:
            print("XGBoost nativo falhou:")
            traceback.print_exc()
    if cfg.rodar_sementes:
        crono.etapa("8d. Estabilidade entre sementes (opcional)")
        try:
            _estabilidade_sementes(cfg, tab, salvos, treino, teste, y_te, saida)
        except Exception:
            print("Teste de sementes falhou:")
            traceback.print_exc()
    if extras:
        mostrar(pd.DataFrame(extras).round(3))

    # ------------------------------------------------------------------ 9. Bootstrap pareado
    crono.etapa("9. Bootstrap pareado")
    boot = av_.bootstrap_pareado(y_te, predicoes, list(tab["Rótulo"]), cfg.n_bootstrap, cfg.seed)
    saida.csv(boot, "bootstrap_pareado_todos_os_pares.csv", index=False)
    print(f"Comparações com o melhor ({melhor}):")
    mostrar(boot[boot["Modelo A"] == melhor].round(3).reset_index(drop=True))
    if len(boot):
        pares_perfil = boot[boot.apply(lambda r: r["Modelo A"].replace(" + perfil", "")
                                       == r["Modelo B"].replace(" + perfil", "")
                                       and r["Modelo A"] != r["Modelo B"], axis=1)]
        print("Efeito do perfil (com vs. sem):")
        mostrar(pares_perfil.round(3).reset_index(drop=True))

    # ------------------------------------------------------------------ 10. Erro desagregado
    crono.etapa("10. Erro por instalação, carga, mês e viés")
    av = teste[["porto", "natureza", "ta", "data_atracacao", "berco_novo"]].copy()
    av["pred"] = predicoes[melhor]
    av["erro"] = av["pred"] - av["ta"]
    av["erro_abs"] = av["erro"].abs()
    av["mes"] = av["data_atracacao"].dt.month

    por_natureza = av_.mae_por(av, "natureza")
    por_porto = av_.mae_por(av, "porto")
    por_mes = av_.mae_por(av, "mes").sort_index()
    por_berco_novo = av_.mae_por(av, "berco_novo")
    for nome_t, t in [("mae_por_tipo_de_carga.csv", por_natureza), ("mae_por_instalacao.csv", por_porto),
                      (f"mae_por_mes_{cfg.ano_teste}.csv", por_mes), ("mae_berco_novo.csv", por_berco_novo)]:
        saida.csv(t, nome_t)
    print("Por tipo de carga:"); mostrar(por_natureza.round(2))
    print("Por instalação (15 maiores):"); mostrar(por_porto.head(15).round(2))
    print(f"Por mês de {cfg.ano_teste} (deriva temporal):"); mostrar(por_mes.round(2))
    print("Berço sem histórico x com histórico:"); mostrar(por_berco_novo.round(2))
    vies = float(av["erro"].mean())
    print(f"Viés médio (predito − observado): {vies:+.2f} h | mediana do erro: {av['erro'].median():+.2f} h")
    top15_share = 100 * por_porto.head(15)["n"].sum() / len(teste)

    saida.fig(av_.fig_mae_porto_carga(por_porto, por_natureza, melhor, cfg.ano_teste), "fig_mae_porto_tipo_carga.png")
    saida.fig(av_.fig_mes_e_residuos(por_mes, av, cfg.ano_teste), "fig_mes_e_residuos.png")
    pd.DataFrame({"id_atracacao": teste["id_atracacao"], "ta": y_te, **predicoes}) \
        .to_csv(saida.caminho("predicoes_teste.csv"), index=False)

    # ------------------------------------------------------------------ 11. SHAP
    crono.etapa("11. SHAP")
    try:
        from .explicabilidade import calcular_shap
        imp_tab, fig = calcular_shap(modelo_melhor, nome_melhor, usa_perfil_melhor, teste, treino, cfg, armador_fonte)
        saida.csv(imp_tab, "shap_importancia.csv")
        mostrar(imp_tab.round(3))
        fig.axes[0].set_title(f"SHAP — {melhor}")
        saida.fig(fig, "fig_shap_beeswarm.png")
    except ImportError:
        print("Pacote 'shap' não instalado: etapa ignorada.")
    except Exception:
        print("SHAP falhou (o resto dos resultados está salvo):")
        traceback.print_exc()

    # ------------------------------------------------------------------ 12. Exportação para o app
    crono.etapa("12. Exportação")
    bercos = treino.groupby("berco_id").agg(porto=("porto", "first"), n=("ta", "size")).reset_index()
    bundle = {
        "nome": melhor, "tipo": nome_melhor, "modelo": modelo_melhor, "usa_perfil": usa_perfil_melhor,
        "cat_cols": CAT_COLS, "time_cols": TIME_COLS, "num_cols": NUM_COLS,
        "perfil_cols": perfil_cols, "frac_cols": frac_cols, "scaler_perfil": perf["scaler"], "kmeans": perf["kmeans"],
        "ref_tabelas": {nome: t.to_dict() for nome, t in ref.tabelas.items()}, "ref_global": ref.global_,
        "bercos": bercos.to_dict("records"),
        "tipos_navegacao": sorted(treino["tipo_navegacao"].unique().tolist()),
        "naturezas": sorted(treino["natureza"].unique().tolist()),
        "armadores": treino["armador"].value_counts().index.tolist(),
        "armador_fonte": armador_fonte,
        "mae_por_porto": por_porto["MAE"].to_dict(),
        "metricas_teste": tab.loc[tab["Rótulo"] == melhor, ["MAE", "RMSE", "R2"]].iloc[0].to_dict(),
        "mae_referencia": mae_ref,
        "periodo": {"treino": cfg.anos_treino, "teste": cfg.ano_teste},
    }
    joblib.dump(bundle, saida.caminho("modelo_streamlit.joblib"), compress=3)

    # ------------------------------------------------------------------ 13. Resumo
    crono.etapa("13. VALORES PARA O ARTIGO")
    m = bundle["metricas_teste"]
    sem_perfil = tab[tab["Perfil K-Means"] == "sem"].set_index("Modelo")
    ensembles = [n for n in ("CatBoost", "Random Forest", "XGBoost") if n in sem_perfil.index]
    b_sem = boot[(boot["Modelo A"] == melhor) & (~boot["Modelo B"].str.contains("perfil"))] if len(boot) else boot
    resumo = {
        "Origem do TA": "TAtracado (TemposAtracacao)" if usa_tempos else "Data Desatracação − Data Atracação",
        "Conferência do TA": conf_ta,
        "Atributo de armador": armador_fonte,
        "Funil": {k: int(v) for k, v in contagem.items()},
        "Corte de extremos (h)": round(ta_max_h, 1) if ta_max_h else None,
        "Corte aplicado a": "treino e teste" if cfg.corte_no_teste else "somente treino",
        "Atracações no teste acima do corte": int((~mask_trunc).sum()),
        "Volume final": f"{len(treino) + len(teste):,} ({len(treino):,} treino; {len(teste):,} teste)",
        "TA conjunto final (mediana/média/dp/Q1/Q3/máx)": [round(float(x), 1) for x in
            [base_final["ta"].median(), base_final["ta"].mean(), base_final["ta"].std(),
             base_final["ta"].quantile(.25), base_final["ta"].quantile(.75), base_final["ta"].max()]],
        "TA mediano por natureza": ta_med_nat.to_dict(),
        "Instalações / berços distintos": f"{n_instal} / {n_bercos}",
        "Berços sem histórico no teste": f"{int(teste['berco_novo'].sum())} atracações "
                                         f"({100 * teste['berco_novo'].mean():.2f}%)",
        "Cascata da referência (%)": cascata["percentual"].to_dict(),
        "k de maior silhueta / k usado / silhueta": f"{perf['k_silhueta']} / {k_usado} / {perf['silhueta']:.2f}",
        "Melhor modelo": melhor,
        "MAE / RMSE / R² (melhor, teste completo)": f"{m['MAE']:.2f} h / {m['RMSE']:.2f} h / {m['R2']:.3f}",
        "MAE referência (teste completo)": f"{mae_ref:.2f} h",
        "Redução vs. referência": f"{100 * (1 - m['MAE'] / mae_ref):.1f}%",
        "Bootstrap vs. variantes SEM perfil": {
            r["Modelo B"]: f"{r['MAE(B) − MAE(A) (h)']:.2f} h (IC {r['IC95% inf']:.2f} a {r['IC95% sup']:.2f})"
            for _, r in b_sem.iterrows()},
        "Diferença 1º−último ensemble (h)": (round(float(sem_perfil.loc[ensembles, "MAE"].max()
                                                         - sem_perfil.loc[ensembles, "MAE"].min()), 2)
                                             if len(ensembles) > 1 else None),
        "Ablação": {r["Variante"]: f"MAE {r['MAE']:.2f} h" for r in ablacao} if ablacao else "não rodada",
        "Viés médio (h)": round(vies, 2),
        "MAE por mês (min–máx, h)": f"{por_mes['MAE'].min():.2f}–{por_mes['MAE'].max():.2f}",
        "15 maiores instalações: % do teste": f"{top15_share:.1f}%",
        "Tempo total (min)": round(crono.minutos(), 1),
    }
    for k, v in resumo.items():
        print(f"{k:45s}: {v}")
    saida.json(resumo, "resumo_para_artigo.json")
    print(f"\nFIM. Resultados em {os.path.abspath(cfg.dir_saida)} (resumo em resumo_para_artigo.json).")
    return resumo


def _versoes():
    v = {"python": platform.python_version()}
    for pacote in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "xgboost", "shap", "matplotlib"):
        try:
            v[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            v[pacote] = None
    return v


# ------------------------------------------------------------------ Experimentos opcionais
def _xgb_nativo(cfg, treino, teste, y_te, predicoes, k_usado):
    from xgboost import XGBRegressor

    cats_ref = {c: sorted(treino[c].unique()) for c in CAT_COLS}

    def matriz_cat(d):
        X = mdl.matriz(d, False).copy()
        for c in CAT_COLS:
            X[c] = pd.Categorical(X[c].where(X[c].isin(cats_ref[c])), categories=cats_ref[c])  # nova -> ausente
        return X

    arq = mdl.arquivo_modelo(cfg.dir_modelos, "xgboost nativo")
    ass = mdl.assinatura(cfg, k_usado, len(treino), extra="xgb_nativo")
    salvo = mdl.carregar_se_valido(arq, ass)
    if salvo is None:
        est = XGBRegressor(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                           n_jobs=-1, random_state=cfg.seed, verbosity=0)
        _, dist = mdl.criar_modelo("XGBoost", False, seed=cfg.seed)
        dist = {k.replace("modelo__", ""): v for k, v in dist.items()}
        busca = RandomizedSearchCV(est, dist, n_iter=cfg.n_iter, cv=TimeSeriesSplit(n_splits=cfg.n_splits_cv),
                                   scoring=mdl.scorer_mae_horas, random_state=cfg.seed, n_jobs=1, refit=True)
        busca.fit(matriz_cat(treino), mdl.para_log(treino["ta"].values))
        salvo = {"modelo": busca.best_estimator_, "params": busca.best_params_, "cv_mae": -busca.best_score_,
                 "assinatura": ass}
        joblib.dump(salvo, arq)
    p = mdl.para_horas(salvo["modelo"].predict(matriz_cat(teste)))
    predicoes["XGBoost (categóricas nativas)"] = p
    return {"Experimento": "XGBoost categóricas nativas", **mdl.metricas(y_te, p)}


def _estabilidade_sementes(cfg, tab, salvos, treino, teste, y_te, saida):
    candidatos = tab[(tab["Perfil K-Means"] == "sem") & (tab["Modelo"] != "Ridge")]["Modelo"].tolist()[:2]
    linhas = []
    for nome in candidatos:
        for s in cfg.sementes_extras:
            modelo, _ = mdl.criar_modelo(nome, False, seed=s)
            modelo.set_params(**salvos[nome]["params"])
            modelo.fit(mdl.matriz(treino, False), mdl.para_log(treino["ta"].values), **mdl.fit_kwargs(nome, False))
            mae_s = mean_absolute_error(y_te, mdl.para_horas(modelo.predict(mdl.matriz(teste, False))))
            linhas.append({"Modelo": nome, "Semente": s, "MAE": mae_s})
            print(f"  {nome} semente {s}: MAE {mae_s:.3f} h", flush=True)
    tab_s = pd.DataFrame(linhas)
    resumo_s = tab_s.groupby("Modelo")["MAE"].agg(["mean", "std", "min", "max"])
    saida.csv(tab_s, "estabilidade_sementes.csv", index=False)
    mostrar(resumo_s.round(3))
    if len(resumo_s) >= 2:
        a, b = resumo_s.index[:2]
        sobrepoe = (resumo_s.loc[a, "max"] >= resumo_s.loc[b, "min"]
                    and resumo_s.loc[b, "max"] >= resumo_s.loc[a, "min"])
        print("  As faixas de MAE " + ("SE SOBREPÕEM: o ordenamento entre os dois não é estável." if sobrepoe
                                       else "não se sobrepõem: o ordenamento é estável entre sementes."))
