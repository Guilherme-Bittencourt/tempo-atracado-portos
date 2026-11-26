"""Testa o núcleo do protótipo (app/nucleo.py) com o modelo exportado por uma execução sintética."""
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from tempo_atracado import Config, executar

from dados_sinteticos import gerar

import nucleo as nc


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    gerar(str(tmp / "dados"))
    cfg = Config(dir_dados=str(tmp / "dados"), dir_saida=str(tmp / "resultados"), modelos=["Ridge"],
                 n_iter=2, n_bootstrap=20, n_shap=100, k_max=4, rodar_ablacao=False,
                 rodar_sensibilidade_corte=False)
    executar(cfg)
    return nc.carregar_bundle(str(tmp / "resultados" / "modelo_streamlit.joblib"))


def _entrada(b, **kw):
    porto = nc.portos(b)[0]
    base = {"porto": porto, "berco_id": nc.bercos_do_porto(b, porto)["berco_id"].iloc[0],
            "tipo_navegacao": b["tipos_navegacao"][0], "natureza": b["naturezas"][0],
            "armador": b["armadores"][0], "mes": 6, "peso_total": 20_000.0,
            "teu_total": 0, "n_registros_carga": 1}
    base.update(kw)
    return pd.DataFrame([base])


def test_predicao_individual(bundle):
    r = nc.prever(bundle, _entrada(bundle)).iloc[0]
    assert 0 < r["ta_previsto"] < 1000
    assert r["ta_referencia"] > 0
    assert r["nivel_referencia"] == "berço × tipo de carga"
    assert not r["berco_sem_historico"]


def test_berco_sem_historico_cai_na_cascata(bundle):
    d = _entrada(bundle, berco_id=None)
    d["berco"] = "Berço inexistente"
    r = nc.prever(bundle, d).iloc[0]
    assert r["berco_sem_historico"]
    assert r["nivel_referencia"] == "porto"


def test_mais_carga_nao_reduz_o_tempo(bundle):
    """O TA sintético cresce com o peso: a predição deve acompanhar."""
    leve = nc.prever(bundle, _entrada(bundle, peso_total=5_000.0))["ta_previsto"].iloc[0]
    pesado = nc.prever(bundle, _entrada(bundle, peso_total=80_000.0))["ta_previsto"].iloc[0]
    assert pesado > leve


def test_lote_preserva_as_linhas(bundle):
    d = pd.concat([_entrada(bundle), _entrada(bundle, peso_total=50_000.0)], ignore_index=True)
    r = nc.prever(bundle, d)
    assert len(r) == 2 and r["ta_previsto"].notna().all()


def test_formatar_horas():
    assert nc.formatar_horas(24.0) == "24h00 (1,0 dia)"
    assert nc.formatar_horas(36.5) == "36h30 (1,5 dias)"


def test_intervalo_empirico():
    """A faixa de calibração devolvida contém a previsão e ordena os quantis observados."""
    import numpy as np
    rng = np.random.default_rng(0)
    pred = rng.uniform(5, 200, 2000)
    d = pd.DataFrame({"Modelo": pred, "ta": pred * rng.lognormal(0, 0.3, len(pred))})
    cal = nc.tabela_calibracao(d, "Modelo")
    assert len(cal) == 10 and cal["n"].sum() == len(d)
    f = nc.intervalo_empirico(cal, 100.0)
    assert f["pred_min"] <= 100.0 <= f["pred_max"]
    assert f["obs_p10"] <= f["obs_mediana"] <= f["obs_p90"]
    assert nc.intervalo_empirico(cal, 10_000.0)["pred_max"] == cal["pred_max"].max()   # acima do máximo
