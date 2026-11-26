"""Teste de ponta a ponta com dados sintéticos (roda em ~1–3 min)."""
import importlib.util
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from tempo_atracado import Config, executar
from tempo_atracado.config import MODELOS_PADRAO

from dados_sinteticos import gerar

DEPENDENCIA = {"CatBoost": "catboost", "XGBoost": "xgboost"}
DISPONIVEIS = [m for m in MODELOS_PADRAO if importlib.util.find_spec(DEPENDENCIA.get(m, "sklearn"))]


def _cfg(tmp_path, **kw):
    base = dict(dir_dados=str(tmp_path / "dados"), dir_saida=str(tmp_path / "resultados"),
                modelos=DISPONIVEIS, n_iter=2, n_bootstrap=50, n_shap=200, k_max=4)
    base.update(kw)
    return Config(**base)


def test_pipeline_completo(tmp_path):
    gerar(str(tmp_path / "dados"))
    cfg = _cfg(tmp_path)
    resumo = executar(cfg)

    for nome in ["funil_preparacao.csv", "tabela_resultados_teste_completo.csv",
                 "tabela_resultados_teste_truncado.csv", "bootstrap_pareado_todos_os_pares.csv",
                 "ablacao_volume.csv", "mae_por_instalacao.csv", "modelo_streamlit.joblib",
                 "resumo_para_artigo.json", "fig_kmeans_k.png", "fig_mae_porto_tipo_carga.png",
                 "utilidade_operacional.csv", "erro_por_faixa_ta.csv",
                 "parecer/ablacao_todos_os_modelos.csv", "parecer/sensibilidade_corte.csv",
                 "parecer/diagnostico_mediana.csv", "parecer/predicoes_teste_completas.csv"]:
        assert os.path.exists(os.path.join(cfg.dir_saida, nome)), nome

    # Ablação em todos os modelos: três variantes por modelo
    abl = pd.read_csv(os.path.join(cfg.dir_saida, "parecer", "ablacao_todos_os_modelos.csv"))
    assert len(abl) == 3 * len(cfg.modelos)
    assert set(abl["Variante"]) == {"completo", "sem peso e TEU", "sem peso, TEU e nº de registros"}
    # Sensibilidade ao corte: uma linha por corte; o corte principal reproduz o modelo principal
    sens = pd.read_csv(os.path.join(cfg.dir_saida, "parecer", "sensibilidade_corte.csv"))
    assert len(sens) == len(cfg.cortes_sensibilidade)
    principal = sens[sens["Corte"] == f"quantil {100 * cfg.ta_quantil_max:.1f}%"].iloc[0]
    assert abs(principal["ΔMAE vs. corte principal (h)"]) < 1e-9
    # SHAP sem conversão em efeito percentual
    shap_tab = os.path.join(cfg.dir_saida, "shap_importancia.csv")
    if os.path.exists(shap_tab):
        assert not any("efeito proporcional" in c for c in pd.read_csv(shap_tab).columns)

    with open(os.path.join(cfg.dir_saida, "resumo_para_artigo.json"), encoding="utf-8") as fh:
        assert json.load(fh)["Melhor modelo"] == resumo["Melhor modelo"]
    assert resumo["Origem do TA"].startswith("Data Desatracação")   # padrão do artigo
    assert resumo["Corte aplicado a"] == "somente treino"
    # Navegação interior é descartada
    funil = resumo["Funil"]
    assert funil["4_longo_curso_e_cabotagem"] < funil["3_com_registro_de_carga"]
    # O TA sintético depende do peso e do berço: o melhor modelo deve bater a referência
    assert float(resumo["Redução vs. referência"].rstrip("%")) > 0


def test_origem_ta_antaq(tmp_path):
    gerar(str(tmp_path / "dados"))
    cfg = _cfg(tmp_path, modelos=["Ridge"], rodar_ablacao=False, rodar_sensibilidade_corte=False,
               origem_ta="antaq")
    resumo = executar(cfg)
    assert resumo["Origem do TA"].startswith("TAtracado")
    assert resumo["Conferência do TA"]["divergencia_maior_que_0_1h"] == 0


def test_reaproveita_modelos(tmp_path, capsys):
    gerar(str(tmp_path / "dados"))
    cfg = _cfg(tmp_path, modelos=["Ridge"], rodar_ablacao=False, rodar_sensibilidade_corte=False)
    executar(cfg)
    capsys.readouterr()
    executar(cfg)
    saida = capsys.readouterr().out
    assert "[Ridge] carregado" in saida and "[Ridge] treinando" not in saida
