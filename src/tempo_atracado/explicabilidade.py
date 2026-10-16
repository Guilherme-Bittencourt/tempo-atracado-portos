"""Importância dos atributos por valores SHAP."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TIME_COLS, NUM_COLS
from .modelos import colunas, matriz


def nomes_pt(armador_fonte):
    return {"log_peso": "Peso bruto (log)", "log_teu": "TEU (log)",
            "log_registros_carga": "Nº de registros de carga (log)",
            "berco_id": "Berço", "porto": "Instalação portuária", "tipo_navegacao": "Tipo de navegação",
            "natureza": "Natureza da carga", "mes": "Mês", "perfil": "Perfil (K-Means)",
            "armador": "Armador (" + ("empresa" if armador_fonte.startswith("empresa") else "nacionalidade") + ")"}


def calcular_shap(modelo, nome_modelo, usa_perfil, teste, treino, cfg, armador_fonte):
    """Retorna (tabela de importância, figura beeswarm)."""
    import shap

    amostra = teste.sample(min(cfg.n_shap, len(teste)), random_state=cfg.seed)
    X_am = matriz(amostra, usa_perfil)
    cats = colunas(usa_perfil)[0]
    if nome_modelo == "CatBoost":
        from catboost import Pool
        sv = modelo.get_feature_importance(Pool(X_am, cat_features=cats), type="ShapValues")[:, :-1]
        X_show = X_am.copy()
        for c in cats:
            X_show[c] = X_show[c].astype("category").cat.codes
    elif nome_modelo in ("Random Forest", "XGBoost"):
        prep, reg = modelo.named_steps["prep"], modelo.named_steps["modelo"]
        X_show = pd.DataFrame(prep.transform(X_am), columns=cats + TIME_COLS + NUM_COLS)
        if nome_modelo == "Random Forest":
            X_show = X_show.iloc[:500]          # TreeExplainer em floresta é lento
        sv = shap.TreeExplainer(reg).shap_values(X_show)
    else:
        prep, reg = modelo.named_steps["prep"], modelo.named_steps["modelo"]
        denso = lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m)
        fundo = denso(prep.transform(matriz(treino.sample(min(2000, len(treino)), random_state=cfg.seed), usa_perfil)))
        Xt = denso(prep.transform(X_am))
        sv = shap.LinearExplainer(reg, fundo).shap_values(Xt)
        X_show = pd.DataFrame(Xt, columns=prep.get_feature_names_out())

    traduz = nomes_pt(armador_fonte)
    nomes = [traduz.get(c, c) for c in X_show.columns]
    imp = pd.Series(np.abs(sv).mean(0), index=nomes).sort_values(ascending=False)
    tab = imp.rename("média |SHAP| (log)").to_frame()
    tab["efeito proporcional médio (%) = exp(média|SHAP|) − 1"] = 100 * (np.exp(tab.iloc[:, 0]) - 1)
    tab["participação na soma (%) — soma 100"] = 100 * tab.iloc[:, 0] / tab.iloc[:, 0].sum()

    shap.summary_plot(sv, X_show.values, feature_names=nomes, max_display=15, show=False)
    fig = plt.gcf()
    try:   # tradução dos rótulos; se a versão do SHAP mudar o layout, a figura sai mesmo assim
        fig.axes[0].set_xlabel("Valor SHAP (impacto na predição, escala log)")
        if len(fig.axes) > 1:
            cb = fig.axes[-1]
            cb.set_ylabel("Valor do atributo")
            if len(cb.get_yticklabels()) == 2:
                cb.set_yticklabels(["Baixo", "Alto"])
    except Exception:
        print("Aviso: não foi possível traduzir os rótulos da figura SHAP.")
    return tab, fig
