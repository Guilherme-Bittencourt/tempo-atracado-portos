"""Protótipo de apoio à decisão: estimativa do tempo atracado (TA) de uma atracação.

Executar na raiz do repositório:

    streamlit run app/app.py

Requer `resultados/modelo_streamlit.joblib`, gerado por `python -m tempo_atracado`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucleo as nc  # noqa: E402

st.set_page_config(page_title="Tempo atracado — protótipo", page_icon="⚓", layout="wide")

TITULO = "Predição do tempo atracado de navios"
SUBTITULO = ("Protótipo de apoio à decisão do Trabalho de Conclusão de Curso — "
             "Centro Universitário Jorge Amado (UNIJORGE), 2026")


@st.cache_resource(show_spinner="Carregando o modelo…")
def _bundle(caminho: str, _mtime: float):
    return nc.carregar_bundle(caminho)


def carregar():
    caminho = nc.caminho_padrao()
    if not os.path.exists(caminho):
        st.error(f"Modelo não encontrado em `{caminho}`.")
        st.markdown(
            "O arquivo é gerado na última etapa do experimento:\n\n"
            "```bash\npip install -e .\n# arquivos da ANTAQ em dados/\npython -m tempo_atracado\n```\n\n"
            "Se o experimento rodou no Colab, baixe `resultados/modelo_streamlit.joblib` e coloque-o "
            "na pasta `resultados/` do repositório, ou aponte a variável de ambiente `MODELO_TA` "
            "para o arquivo.")
        st.stop()
    return _bundle(caminho, os.path.getmtime(caminho))


@st.cache_data(show_spinner=False)
def _calibracao(nome_modelo: str):
    """Faixas de predição do teste e o TA observado em cada uma (lê resultados/predicoes_teste.csv)."""
    pred = nc.ler_csv_resultado("predicoes_teste.csv")
    if pred is None or nome_modelo not in pred.columns:
        return None
    return nc.tabela_calibracao(pred, nome_modelo)


def _num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


# ------------------------------------------------------------------ páginas
def pagina_predicao(b: dict):
    st.subheader("Estimativa para uma atracação")
    st.caption("Informe as condições da atracação programada. Peso bruto, TEU e nº de registros de carga "
               "só são conhecidos ao término da operação: aqui entram como **estimativa prévia** "
               "(por exemplo, do manifesto). O desempenho do modelo com volume estimado **não foi "
               "verificado** — veja *Sobre o modelo*.")

    col1, col2, col3 = st.columns(3)
    with col1:
        porto = st.selectbox("Instalação portuária", nc.portos(b))
        bercos = nc.bercos_do_porto(b, porto)
        berco_id = st.selectbox("Berço", bercos["berco_id"].tolist(), format_func=nc.rotulo_berco,
                                help="Ordenados pelo nº de atracações no treino.")
        n_hist = int(bercos.loc[bercos["berco_id"] == berco_id, "n"].iloc[0])
        st.caption(f"{n_hist:,} atracações desse berço no treino.".replace(",", "."))
    with col2:
        natureza = st.selectbox("Natureza predominante da carga", b["naturezas"])
        tipo_nav = st.selectbox("Tipo de navegação", b["tipos_navegacao"])
        armador = st.selectbox(f"Armador ({b['armador_fonte']})", b["armadores"])
    with col3:
        mes = st.selectbox("Mês previsto da atracação", list(nc.MESES), format_func=nc.MESES.get,
                           index=pd.Timestamp.today().month - 1)
        peso = st.number_input("Peso bruto estimado (t)", min_value=0.0, value=20_000.0, step=1_000.0)
        teu = st.number_input("TEU estimado", min_value=0, value=0, step=50)
        registros = st.number_input("Nº de registros de carga", min_value=1, value=1, step=1,
                                    help="Quantidade de lançamentos de carga da atracação no "
                                         "Estatístico Aquaviário — indica o fracionamento da operação.")

    entrada = pd.DataFrame([{"porto": porto, "berco_id": berco_id, "tipo_navegacao": tipo_nav,
                             "natureza": natureza, "armador": armador, "mes": mes,
                             "peso_total": peso, "teu_total": teu, "n_registros_carga": registros}])
    r = nc.prever(b, entrada).iloc[0]

    st.divider()
    previsto, ref, mae = float(r["ta_previsto"]), float(r["ta_referencia"]), float(r["mae_instalacao"])
    c1, c2, c3 = st.columns(3)
    c1.metric(f"TA previsto ({b['nome']})", nc.formatar_horas(previsto))
    c2.metric("Referência ingênua (mediana histórica)", nc.formatar_horas(ref),
              delta=f"{previsto - ref:+.1f}".replace(".", ",") + " h vs. o previsto", delta_color="off")
    c3.metric("MAE desta instalação no teste", f"{_num(mae)} h",
              help=f"Erro absoluto médio do modelo nas atracações de {b['periodo']['teste']} desta instalação, "
                   "com o volume realizado. É uma média, não um limite de erro.")
    st.caption(f"A referência usou o nível *{r['nivel_referencia']}* da cascata.")

    cal = _calibracao(b["nome"])
    if cal is not None:
        f = nc.intervalo_empirico(cal, previsto)
        st.info(
            f"**O que aconteceu em {b['periodo']['teste']} com previsões parecidas.** Nas "
            f"{int(f['n']):,}".replace(",", ".")
            + f" atracações do teste com previsão entre {_num(f['pred_min'])} h e {_num(f['pred_max'])} h, "
            + f"80% dos tempos observados ficaram entre **{_num(f['obs_p10'])} h** e **{_num(f['obs_p90'])} h** "
              f"(mediana de {_num(f['obs_mediana'])} h), e o erro ficou dentro de ±12 h em "
              f"{_num(f['pct_dentro_12h'], 0)}% delas. Esses valores usam o volume realizado; com volume "
              "estimado, a incerteza tende a ser maior.")
        if f["pct_acima_120h"] >= 10:
            st.warning(f"Nessa faixa de previsão, {_num(f['pct_acima_120h'], 0)}% das atracações de "
                       f"{b['periodo']['teste']} passaram de 120 h. O modelo **subestima sistematicamente as "
                       "estadias longas**: trate a estimativa como valor central, não como teto.")
    if r["berco_sem_historico"]:
        st.warning("Berço sem histórico no conjunto de treino: a estimativa é menos confiável.")
    if b["usa_perfil"] and "perfil" in r:
        st.caption(f"Perfil operacional (K-Means): **{r['perfil']}**.")

    st.markdown("##### Sensibilidade ao volume estimado")
    st.caption("Como a estimativa muda se o peso bruto do manifesto variar — o atributo mais "
               "influente segundo o SHAP.")
    fatores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
    cenarios = pd.concat([entrada.assign(peso_total=peso * f, teu_total=int(teu * f)) for f in fatores],
                         ignore_index=True)
    sens = nc.prever(b, cenarios)
    st.line_chart(sens.set_index("peso_total")["ta_previsto"], y_label="TA previsto (h)",
                  x_label="Peso bruto estimado (t)", height=280)
    faixa = sens["ta_previsto"]
    st.caption("Entre metade e o dobro do peso informado, a estimativa vai de "
               f"{faixa.min():.1f} h a {faixa.max():.1f} h".replace(".", ",") + ".")


def pagina_lote(b: dict):
    st.subheader("Estimativa em lote")
    st.caption("Envie a programação de atracações em CSV e baixe as estimativas. "
               "Útil para comparar janelas de atracação de um dia ou de uma semana.")

    modelo_csv = pd.DataFrame([
        {"porto": nc.portos(b)[0], "berco": nc.rotulo_berco(nc.bercos_do_porto(b, nc.portos(b)[0])["berco_id"].iloc[0]),
         "tipo_navegacao": b["tipos_navegacao"][0], "natureza": b["naturezas"][0],
         "armador": b["armadores"][0], "mes": 1, "peso_total": 20000, "teu_total": 0, "n_registros_carga": 1}])
    st.download_button("Baixar CSV de exemplo", modelo_csv.to_csv(index=False).encode("utf-8"),
                       "exemplo_programacao.csv", "text/csv")

    arquivo = st.file_uploader("Arquivo CSV", type="csv")
    if arquivo is None:
        st.caption(f"Colunas esperadas: {', '.join(nc.ENTRADAS)}.")
        return

    d = pd.read_csv(arquivo)
    faltando = [c for c in nc.ENTRADAS if c not in d.columns and not (c == "berco" and "berco_id" in d.columns)]
    if faltando:
        st.error(f"Faltam colunas no arquivo: {', '.join(faltando)}.")
        return

    r = nc.prever(b, d)
    st.success(f"{len(r)} atracações estimadas.")
    colunas = ["porto", "berco_id", "natureza", "mes", "peso_total",
               "ta_previsto", "ta_referencia", "mae_instalacao", "berco_sem_historico"]
    st.dataframe(r[colunas].round(2), width="stretch", hide_index=True)
    st.download_button("Baixar estimativas", r[colunas].to_csv(index=False).encode("utf-8"),
                       "estimativas_ta.csv", "text/csv")
    st.metric("TA previsto — soma da programação", nc.formatar_horas(float(r["ta_previsto"].sum())))


def pagina_desempenho(b: dict):
    st.subheader("Desempenho do modelo")
    m = b["metricas_teste"]
    periodo = b["periodo"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{m['MAE']:.2f} h".replace(".", ","))
    c2.metric("RMSE", f"{m['RMSE']:.2f} h".replace(".", ","))
    c3.metric("R²", f"{m['R2']:.3f}".replace(".", ","))
    c4.metric("Redução vs. referência", f"{100 * (1 - m['MAE'] / b['mae_referencia']):.1f}%".replace(".", ","))
    st.caption(f"Modelo **{b['nome']}** treinado em {'/'.join(map(str, periodo['treino']))} e avaliado "
               f"nas atracações de {periodo['teste']}, com o volume de carga realizado.")

    tab = nc.ler_csv_resultado("tabela_resultados_teste_completo.csv")
    if tab is not None:
        st.markdown("##### Comparação entre algoritmos")
        st.dataframe(tab.drop(columns=["Rótulo"], errors="ignore").round(3),
                     width="stretch", hide_index=True)

    st.markdown("##### Dependência dos atributos de volume")
    st.caption("Peso bruto, TEU e nº de registros de carga só existem na base ao término da operação. "
               "A tabela mostra o MAE (h) e a redução sobre a referência ingênua quando esses atributos são "
               "retirados do modelo, com os hiperparâmetros do modelo completo. Valores negativos: erro maior "
               "que o da referência.")
    abl_todos = nc.ler_csv_resultado(os.path.join("parecer", "ablacao_todos_os_modelos.csv"))
    abl = nc.ler_csv_resultado("ablacao_volume.csv")
    if abl_todos is not None:
        ordem = ["completo", "sem peso e TEU", "sem peso, TEU e nº de registros"]
        tab_abl = abl_todos.assign(
            celula=[f"{_num(m, 2)} h ({_num(r)}%)" for m, r in zip(abl_todos["MAE"], abl_todos["Redução vs. referência (%)"])])
        st.dataframe(tab_abl.pivot(index="Modelo", columns="Variante", values="celula")[ordem],
                     width="stretch")
    elif abl is not None:
        st.dataframe(abl.round(3), width="stretch", hide_index=True)
    st.caption("Com uma estimativa prévia de volume (como a do manifesto), é razoável supor um desempenho "
               "intermediário entre o modelo completo e as variantes sem volume, mas essa **hipótese não foi "
               "verificada**: depende da existência e da precisão da estimativa, ausente da base pública.")

    faixa = nc.ler_csv_resultado("erro_por_faixa_ta.csv")
    if faixa is not None:
        st.markdown("##### Erro por duração da atracação")
        st.caption("O erro cresce com a duração observada: o modelo superestima as estadias curtas e "
                   "subestima as longas.")
        faixa = faixa.rename(columns={faixa.columns[0]: "TA observado"})
        mostrar = faixa[["TA observado", "pct_do_teste", "MAE", "vies_medio_h", "pct_dentro_12h"]].rename(columns={
            "pct_do_teste": "% do teste", "MAE": "MAE (h)", "vies_medio_h": "Viés (h)",
            "pct_dentro_12h": "% dentro de ±12 h"})
        st.dataframe(mostrar.round(1), width="stretch", hide_index=True)

    shap = nc.ler_csv_resultado("shap_importancia.csv")
    if shap is not None:
        st.markdown("##### Influência dos atributos (SHAP)")
        st.caption("Participação de cada atributo na soma das importâncias (%), em escala logarítmica. "
                   "Indica o peso relativo do atributo nas predições, não um efeito percentual sobre o TA.")
        shap = shap.rename(columns={shap.columns[0]: "Atributo"})
        coluna = ([c for c in shap.columns if "participação" in c.lower()] or [shap.columns[-1]])[0]
        serie = shap.set_index("Atributo")[coluna].sort_values(ascending=False).head(15)
        st.bar_chart(serie, horizontal=True, height=30 * len(serie) + 60)

    porto_tab = nc.ler_csv_resultado("mae_por_instalacao.csv")
    if porto_tab is not None:
        st.markdown("##### Erro por instalação portuária")
        st.caption("MAE em horas, nas instalações de maior movimento no conjunto de teste.")
        n = st.slider("Instalações exibidas", 5, min(40, len(porto_tab)), min(15, len(porto_tab)))
        st.bar_chart(porto_tab.head(n).set_index("porto")["MAE"], horizontal=True, height=30 * n + 60)


def pagina_sobre(b: dict):
    st.subheader("Sobre o modelo")
    m = b["metricas_teste"]
    nota_perfil = ("O perfil operacional obtido por K-Means entra como atributo adicional."
                   if b["usa_perfil"] else
                   "O perfil operacional obtido por K-Means não melhorou o erro e não é usado na predição.")
    st.markdown(f"""
**O que o protótipo faz.** Estima o tempo atracado (TA) — o intervalo entre a atracação e a
desatracação — de uma atracação ainda não realizada, a partir das suas condições operacionais.
A estimativa é apresentada ao lado da referência ingênua (a mediana histórica do tempo atracado por
berço e tipo de carga), de modo que o ganho do modelo fique explícito na tela.

**Cenário de uso.** Estimativa antes da atracação, para apoio à programação dos berços, a partir apenas
de informações da base pública da ANTAQ. O protótipo não usa dados internos dos terminais.

**Modelo em uso.** {b['nome']}, escolhido por ter o menor MAE entre Regressão Ridge, Random Forest,
XGBoost e CatBoost. Treinado com as atracações de
{'/'.join(map(str, b['periodo']['treino']))} e avaliado nas de {b['periodo']['teste']}
(MAE {m['MAE']:.2f} h; R² {m['R2']:.3f}). O alvo é modelado em log(1 + TA); a reconversão para
horas aproxima a mediana condicional do tempo atracado, sem garanti-la.

**Atributos.** Instalação portuária, berço, tipo de navegação, natureza predominante da carga,
{b['armador_fonte']}, mês, peso bruto, TEU e número de registros de carga (os três últimos em log).
{nota_perfil}

**Dados.** Bases de Atracação e Carga do Estatístico Aquaviário da ANTAQ, navegações de longo curso
e cabotagem.

#### Limitações
- **O volume é informado, não previsto.** Peso bruto, TEU e número de registros de carga só existem
  na base pública ao término da operação, e o desempenho mostrado na aba *Desempenho* foi medido com o
  volume realizado. Sem esses atributos, o ganho sobre a referência praticamente desaparece (ver a
  ablação). O desempenho com uma estimativa prévia de volume, como a do manifesto, **não foi
  verificado**. A aba *Predição* permite ver como a estimativa reage a variações do volume informado.
- **Estadias longas.** O modelo subestima sistematicamente as atracações acima de 120 h, mesmo
  quando treinado sem o corte de extremos. Para essas, a estimativa pontual é pouco informativa.
- **Natureza predominante.** A interface pede uma única natureza de carga; no experimento o modelo
  também dispunha das frações de peso por natureza, usadas apenas pelo K-Means. Para atracações com
  cargas muito mistas, a simplificação da interface pode deslocar o perfil atribuído.
- **Berços sem histórico.** Berços ausentes do treino recebem estimativa menos confiável, sinalizada
  na tela.
- **Erro desigual entre instalações.** O MAE varia bastante entre portos e tipos de carga; a faixa
  exibida usa o MAE da própria instalação, quando disponível.
- **Não há previsão de fila.** O modelo estima o tempo atracado, não o tempo de espera por berço,
  e não incorpora condições meteorológicas, greves ou indisponibilidade de equipamento.

#### Como regenerar o modelo
```bash
pip install -e .
# arquivos da ANTAQ (Atracação e Carga, 2022–2024) em dados/
python -m tempo_atracado
streamlit run app/app.py
```
""")


# ------------------------------------------------------------------ navegação
def main():
    b = carregar()
    with st.sidebar:
        st.markdown(f"### ⚓ {TITULO}")
        st.caption(SUBTITULO)
        pagina = st.radio("Seções", ["Predição", "Lote (CSV)", "Desempenho", "Sobre o modelo"],
                          label_visibility="collapsed")
        st.divider()
        st.caption(f"Modelo: **{b['nome']}**  \nMAE {_num(b['metricas_teste']['MAE'], 2)} h · "
                   f"R² {_num(b['metricas_teste']['R2'], 3)}")

    st.title(TITULO)
    {"Predição": pagina_predicao, "Lote (CSV)": pagina_lote,
     "Desempenho": pagina_desempenho, "Sobre o modelo": pagina_sobre}[pagina](b)


if __name__ == "__main__":
    main()
