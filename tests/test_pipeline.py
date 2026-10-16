"""Teste de ponta a ponta com dados sintéticos (roda em ~1–3 min)."""
import importlib.util
import json
import os

import matplotlib

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
                 "resumo_para_artigo.json", "fig_kmeans_k.png", "fig_mae_porto_tipo_carga.png"]:
        assert os.path.exists(os.path.join(cfg.dir_saida, nome)), nome

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
    cfg = _cfg(tmp_path, modelos=["Ridge"], rodar_ablacao=False, origem_ta="antaq")
    resumo = executar(cfg)
    assert resumo["Origem do TA"].startswith("TAtracado")
    assert resumo["Conferência do TA"]["divergencia_maior_que_0_1h"] == 0


def test_reaproveita_modelos(tmp_path, capsys):
    gerar(str(tmp_path / "dados"))
    cfg = _cfg(tmp_path, modelos=["Ridge"], rodar_ablacao=False)
    executar(cfg)
    capsys.readouterr()
    executar(cfg)
    saida = capsys.readouterr().out
    assert "[Ridge] carregado" in saida and "[Ridge] treinando" not in saida
