"""Definição, busca de hiperparâmetros e cache dos modelos de regressão.

O alvo é modelado em log(1 + TA); a busca de hiperparâmetros minimiza o MAE em horas.
CatBoost e XGBoost são importados só quando usados.
"""
import hashlib
import json
import os
import re
import time
import unicodedata

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .config import CAT_COLS, NUM_COLS, TIME_COLS


def colunas(usa_perfil, sem=()):
    cats = CAT_COLS + (["perfil"] if usa_perfil else [])
    return cats, TIME_COLS, [c for c in NUM_COLS if c not in sem]


def matriz(d, usa_perfil, sem=()):
    cats, tempo, nums = colunas(usa_perfil, sem)
    return d[cats + tempo + nums]


def fit_kwargs(nome, usa_perfil):
    return {"cat_features": colunas(usa_perfil)[0]} if nome == "CatBoost" else {}


def criar_modelo(nome, usa_perfil, sem=(), seed=42):
    """Retorna (estimador, espaço de busca)."""
    cats, tempo, nums = colunas(usa_perfil, sem)
    if nome == "Ridge":
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats + tempo),
                                  ("num", StandardScaler(), nums)])
        return Pipeline([("prep", prep), ("modelo", Ridge())]), {"modelo__alpha": loguniform(1e-3, 1e3)}
    if nome in ("Random Forest", "XGBoost"):
        prep = ColumnTransformer([("cat", TargetEncoder(target_type="continuous", random_state=seed), cats),
                                  ("num", "passthrough", tempo + nums)])
        if nome == "Random Forest":
            reg = RandomForestRegressor(n_jobs=-1, random_state=seed)
            dist = {"modelo__n_estimators": randint(100, 400), "modelo__max_depth": [None, 12, 18, 24, 32],
                    "modelo__min_samples_leaf": randint(1, 20), "modelo__max_features": [0.3, 0.5, 0.7, 1.0]}
        else:
            from xgboost import XGBRegressor
            reg = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)
            dist = {"modelo__n_estimators": randint(300, 1500), "modelo__learning_rate": loguniform(0.02, 0.2),
                    "modelo__max_depth": randint(4, 11), "modelo__min_child_weight": randint(1, 11),
                    "modelo__subsample": uniform(0.6, 0.4), "modelo__colsample_bytree": uniform(0.6, 0.4),
                    "modelo__reg_lambda": loguniform(0.1, 10)}
        return Pipeline([("prep", prep), ("modelo", reg)]), dist
    if nome == "CatBoost":
        from catboost import CatBoostRegressor
        est = CatBoostRegressor(loss_function="RMSE", verbose=0, random_seed=seed, thread_count=-1,
                                allow_writing_files=False)
        dist = {"iterations": randint(500, 1500), "learning_rate": loguniform(0.03, 0.2),
                "depth": randint(4, 10), "l2_leaf_reg": loguniform(1, 10)}
        return est, dist
    raise ValueError(f"Modelo desconhecido: {nome}")


def para_log(y):
    return np.log1p(y)


def para_horas(z):
    return np.clip(np.expm1(z), 0, None)


def metricas(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


scorer_mae_horas = make_scorer(lambda zt, zp: mean_absolute_error(para_horas(zt), para_horas(zp)),
                               greater_is_better=False)


# ------------------------------------------------------------------ Cache em disco
def arquivo_modelo(dir_modelos, rotulo):
    s = unicodedata.normalize("NFKD", rotulo.lower())
    s = "".join(c for c in s if not unicodedata.combining(c)).encode("ascii", "ignore").decode()
    return os.path.join(dir_modelos, re.sub(r"\W+", "_", s).strip("_") + ".joblib")


def assinatura(cfg, k, n_treino, extra=""):
    """Hash da configuração que define um modelo treinado (mesmo formato da versão de Colab)."""
    conteudo = {"anos": cfg.anos, "quantil": cfg.ta_quantil_max, "cats": CAT_COLS, "nums": NUM_COLS,
                "n_iter": cfg.n_iter, "cv": cfg.n_splits_cv, "amostra": cfg.amostra_tuning, "seed": cfg.seed,
                "k": k, "n_treino": n_treino, "extra": extra}
    return hashlib.md5(json.dumps(conteudo, sort_keys=True, default=str).encode()).hexdigest()[:10]


def carregar_se_valido(arq, assinatura_atual):
    if not os.path.exists(arq):
        return None
    try:
        salvo = joblib.load(arq)
    except Exception as e:
        print(f"  [{os.path.basename(arq)}] não foi possível carregar ({type(e).__name__}): será retreinado.")
        return None
    ass = salvo.get("assinatura")
    if ass is None:
        print(f"  [{os.path.basename(arq)}] modelo da execução original (sem assinatura): reaproveitado. "
              f"O treino não mudou, só o teste.")
        return salvo
    if ass == assinatura_atual:
        return salvo
    print(f"  [{os.path.basename(arq)}] configuração diferente da atual: será retreinado.")
    return None


def treinar_com_busca(nome, usa_perfil, treino_tun, treino, cfg, ass):
    """Busca aleatória com validação temporal e reajuste no treino completo."""
    est, dist = criar_modelo(nome, usa_perfil, seed=cfg.seed)
    fit_kw = fit_kwargs(nome, usa_perfil)
    busca = RandomizedSearchCV(est, dist, n_iter=cfg.n_iter, cv=TimeSeriesSplit(n_splits=cfg.n_splits_cv),
                               scoring=scorer_mae_horas, random_state=cfg.seed, n_jobs=1, refit=False)
    t0 = time.time()
    busca.fit(matriz(treino_tun, usa_perfil), para_log(treino_tun["ta"].values), **fit_kw)
    t_busca = time.time() - t0
    modelo, _ = criar_modelo(nome, usa_perfil, seed=cfg.seed)
    modelo.set_params(**busca.best_params_)
    t0 = time.time()
    modelo.fit(matriz(treino, usa_perfil), para_log(treino["ta"].values), **fit_kw)
    return {"modelo": modelo, "params": busca.best_params_, "cv_mae": -busca.best_score_,
            "t_busca": t_busca, "t_fit": time.time() - t0, "assinatura": ass}
