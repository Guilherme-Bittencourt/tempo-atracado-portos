"""Localização, leitura e integração das bases do Estatístico Aquaviário da ANTAQ."""
import os
import zipfile

import numpy as np
import pandas as pd

from .utils import mostrar, normalizar

TIPOS_ARQ = {"atracacao": "Atracacao", "temposatracacao": "TemposAtracacao", "carga": "Carga"}
IGNORAR_PASTAS = {"sample_data", "drive", ".git", "__pycache__", ".ipynb_checkpoints"}

# Nome interno -> apelidos possíveis (já normalizados) da coluna no arquivo da ANTAQ
COLUNAS = {
    "atracacao": {
        "id_atracacao": ["idatracacao"],
        "berco": ["berco"],
        "porto": ["portoatracacao"],
        "data_atracacao": ["dataatracacao"],
        "data_desatracacao": ["datadesatracacao"],
        "tipo_navegacao": ["tipodenavegacaodaatracacao"],
        "empresa_navegacao": ["armador", "nomedoarmador", "nomearmador", "empresadenavegacao",
                              "empresanavegacao", "empresa"],
        "nacionalidade_armador": ["nacionalidadedoarmador"],
        "tipo_autoridade": ["tipodaautoridadeportuaria"],
    },
    "temposatracacao": {"id_atracacao": ["idatracacao"], "ta": ["tatracado"]},
    "carga": {
        "id_atracacao": ["idatracacao"],
        "natureza": ["naturezadacarga"],
        "peso": ["vlpesocargabruta"],
        "teu": ["teu"],
        "conteiner_estado": ["conteinerestado"],
        "flag_mc": ["flagmcoperacaocarga"],
    },
}
OPCIONAIS = {"empresa_navegacao", "nacionalidade_armador", "data_desatracacao", "tipo_autoridade",
             "conteiner_estado", "flag_mc"}


# ------------------------------------------------------------------ Arquivos
def _listar(raiz, extensoes, ignorar=()):
    ignorar = {os.path.realpath(p) for p in ignorar}
    achados = []
    for pasta, subpastas, arquivos in os.walk(raiz):
        subpastas[:] = [d for d in subpastas if d not in IGNORAR_PASTAS
                        and os.path.realpath(os.path.join(pasta, d)) not in ignorar]
        achados += [os.path.join(pasta, f) for f in arquivos if f.lower().endswith(extensoes)]
    return sorted(achados)


def extrair_zips(cfg):
    """Extrai os .txt dos zips da ANTAQ e restaura modelos de um resultados.zip antigo, se houver."""
    os.makedirs(cfg.dir_extraidos, exist_ok=True)
    os.makedirs(cfg.dir_modelos, exist_ok=True)
    zips = _listar(cfg.dir_dados, (".zip",), ignorar=[cfg.dir_saida])
    print(f"Zips encontrados: {[os.path.basename(z) for z in zips] or 'nenhum'}")
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                for nome in zf.namelist():
                    base = os.path.basename(nome)
                    if nome.lower().endswith(".txt"):
                        destino = os.path.join(cfg.dir_extraidos, base)
                    elif nome.lower().endswith(".joblib") and "modelos/" in nome.replace("\\", "/"):
                        destino = os.path.join(cfg.dir_modelos, base)   # modelos da execução original
                    else:
                        continue
                    if not os.path.exists(destino):
                        with zf.open(nome) as src, open(destino, "wb") as dst:
                            while True:
                                bloco = src.read(1 << 24)
                                if not bloco:
                                    break
                                dst.write(bloco)
                        if destino.endswith(".joblib"):
                            print(f"  modelo restaurado: {base}")
        except zipfile.BadZipFile:
            print(f"  aviso: zip inválido ignorado: {z}")


def localizar_arquivos(cfg):
    """Retorna {(ano, tipo): caminho} e se o arquivo TemposAtracacao existe para todos os anos."""
    por_nome = {}
    for f in _listar(cfg.dir_dados, (".txt",), ignorar=[cfg.dir_saida]):
        por_nome.setdefault(normalizar(os.path.splitext(os.path.basename(f))[0]), f)
    arquivos = {(a, t): por_nome.get(f"{a}{t}") for a in cfg.anos for t in TIPOS_ARQ}
    for (a, t), f in arquivos.items():
        info = f"{os.path.basename(f)} ({os.path.getsize(f) / 1e6:.1f} MB)" if f else "NÃO ENCONTRADO"
        print(f"  {a}{TIPOS_ARQ[t]:16s} -> {info}")

    obrigatorios = ("atracacao", "carga") + (("temposatracacao",) if cfg.origem_ta == "antaq" else ())
    faltando = [f"{a}{TIPOS_ARQ[t]}" for a in cfg.anos for t in obrigatorios if not arquivos[(a, t)]]
    if faltando:
        raise FileNotFoundError(f"Faltam em '{cfg.dir_dados}': " + ", ".join(faltando)
                                + ". Veja a seção 'Dados' do README.")
    if cfg.origem_ta not in ("datas", "antaq"):
        raise ValueError(f"origem_ta deve ser 'datas' ou 'antaq', não {cfg.origem_ta!r}")
    usa_tempos = cfg.origem_ta == "antaq"
    print("\n>>> ORIGEM DO TA: " + ("coluna TAtracado do arquivo TemposAtracacao da ANTAQ" if usa_tempos else
                                 "Data Desatracação − Data Atracação") + " <<<")
    return arquivos, usa_tempos


# ------------------------------------------------------------------ Leitura
def para_numero(s):
    """Converte texto no padrão brasileiro (1.234,5) para número."""
    s = s.astype(str).str.strip()
    com_virgula = s.str.contains(",", na=False)
    s = s.where(~com_virgula, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def para_data(s):
    d = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if d.isna().mean() > 0.5:
        d = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return d


def ler_tabela(caminho, tipo):
    mapa = COLUNAS[tipo]
    alvo_por_norm = {apelido: destino for destino, apelidos in mapa.items() for apelido in apelidos}
    df = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(caminho, sep=";", dtype=str, encoding=enc, low_memory=False,
                             usecols=lambda c: normalizar(c) in alvo_por_norm)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"Não foi possível ler {caminho} em UTF-8 nem em Latin-1.")
    renomear = {}
    for c in df.columns:
        destino = alvo_por_norm[normalizar(c)]
        if destino not in renomear.values():
            renomear[c] = destino
    df = df[list(renomear)].rename(columns=renomear)
    faltam = [c for c in mapa if c not in df.columns and c not in OPCIONAIS]
    if faltam:
        cab = pd.read_csv(caminho, sep=";", nrows=0, encoding="latin-1").columns.tolist()
        raise KeyError(f"{os.path.basename(caminho)}: colunas {faltam} não encontradas. Disponíveis: {cab}")
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def carregar_bases(cfg, arquivos, usa_tempos, saida):
    """Lê as três bases, define o atributo de armador e a origem do TA."""
    ler = lambda tipo: pd.concat([ler_tabela(arquivos[(a, tipo)], tipo) for a in cfg.anos], ignore_index=True)
    atr = ler("atracacao")
    carga = ler("carga")
    tempos = ler("temposatracacao") if usa_tempos else None
    print(f"Atracações (linhas brutas): {len(atr):,} | Registros de carga: {len(carga):,}")

    if "empresa_navegacao" in atr and atr["empresa_navegacao"].notna().mean() > 0.5:
        atr["armador"] = atr["empresa_navegacao"]
        armador_fonte = "empresa de navegação"
    elif "nacionalidade_armador" in atr:
        atr["armador"] = atr["nacionalidade_armador"]
        armador_fonte = "nacionalidade do armador"
    else:
        raise KeyError("Nenhuma coluna de armador encontrada na base de Atracação.")
    print(f">>> ATRIBUTO DE ARMADOR: {armador_fonte.upper()} <<<")

    atr = atr.drop_duplicates("id_atracacao", keep="last")
    atr["data_atracacao"] = para_data(atr["data_atracacao"])
    if "data_desatracacao" in atr:
        atr["data_desatracacao"] = para_data(atr["data_desatracacao"])
        atr["ta_datas"] = (atr["data_desatracacao"] - atr["data_atracacao"]).dt.total_seconds() / 3600
    if usa_tempos:
        tempos = tempos.drop_duplicates("id_atracacao", keep="last")
        tempos["ta"] = para_numero(tempos["ta"])
    else:
        if "ta_datas" not in atr:
            raise KeyError("Sem TemposAtracacao e sem 'Data Desatracação'.")
        tempos = pd.DataFrame({"id_atracacao": atr["id_atracacao"], "ta": atr["ta_datas"]})
    carga["peso"] = para_numero(carga["peso"])
    carga["teu"] = para_numero(carga["teu"])

    diagnosticar_peso(carga, saida)
    return atr, tempos, carga, armador_fonte


def diagnosticar_peso(carga, saida):
    """Ausência e zeros de peso por natureza da carga (relevante para contêineres, medidos também em TEU)."""
    diag = carga.groupby("natureza").agg(
        registros=("peso", "size"),
        peso_ausente_pct=("peso", lambda s: 100 * s.isna().mean()),
        peso_zero_pct=("peso", lambda s: 100 * (s == 0).mean()),
        peso_mediano_t=("peso", "median"),
        teu_total=("teu", "sum"))
    saida.csv(diag, "diag_peso_por_natureza.csv")
    print("\nDiagnóstico do peso por natureza da carga:")
    mostrar(diag.round(2))
    if "flag_mc" in carga:
        print("Distribuição de FlagMCOperacaoCarga (verifique o significado no dicionário de dados da ANTAQ):")
        print(carga.groupby("flag_mc")["peso"].agg(["size", "sum"]).to_string())
    if "conteiner_estado" in carga:
        print("Estado do contêiner (cheio/vazio) — o peso de vazios entra na soma:")
        print(carga.groupby("conteiner_estado")["peso"].agg(["size", "sum"]).to_string())


# ------------------------------------------------------------------ Integração
def agregar_carga(c):
    """Uma linha por atracação: peso, TEU, nº de registros, natureza predominante e frações por natureza."""
    c = c.copy()
    c["peso_calc"] = c["peso"].fillna(0)
    g = c.groupby("id_atracacao")
    agg = pd.DataFrame({
        "peso_total": g["peso"].sum(min_count=1),
        "teu_total": g["teu"].sum(min_count=1).fillna(0),
        "n_registros_carga": g.size(),
    })
    piv = c.pivot_table(index="id_atracacao", columns="natureza", values="peso_calc", aggfunc="sum", fill_value=0)
    nat = piv.idxmax(axis=1)
    nat[piv.sum(axis=1) <= 0] = np.nan          # sem peso: natureza indefinida
    agg["natureza"] = nat
    frac = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    frac.columns = ["frac_" + normalizar(col) for col in frac.columns]
    return agg.join(frac)


def montar_base(atr, tempos, carga):
    """Junta atracação, tempos e carga agregada e cria os atributos. Retorna (df, contagem do funil)."""
    carga_agg = agregar_carga(carga)
    contagem = {"1_atracacoes_brutas_2022_2024": len(atr)}
    df = atr.merge(tempos[["id_atracacao", "ta"]], on="id_atracacao", how="inner")
    contagem["2_com_registro_de_tempo"] = len(df)
    df = df.merge(carga_agg, left_on="id_atracacao", right_index=True, how="inner")
    contagem["3_com_registro_de_carga"] = len(df)

    df["berco_id"] = df["porto"].astype(str) + " | " + df["berco"].astype(str)
    df["mes"] = df["data_atracacao"].dt.month
    df["ano"] = df["data_atracacao"].dt.year
    df["log_peso"] = np.log1p(df["peso_total"].clip(lower=0))
    df["log_teu"] = np.log1p(df["teu_total"].clip(lower=0))
    df["log_registros_carga"] = np.log1p(df["n_registros_carga"])
    return df, contagem


def filtrar_e_dividir(df, contagem, cfg):
    """Filtros, divisão temporal e corte de extremos (calculado no treino)."""
    nav_ok = {normalizar(n) for n in cfg.navegacoes}
    df = df[df["tipo_navegacao"].map(normalizar).isin(nav_ok)]
    contagem["4_longo_curso_e_cabotagem"] = len(df)
    df = df.dropna(subset=["data_atracacao", "ta", "peso_total", "porto", "berco", "tipo_navegacao",
                           "natureza", "armador"])
    contagem["5_sem_incompletos"] = len(df)
    df = df[(df["ta"] > 0) & (df["peso_total"] > 0)]
    contagem["6_sem_TA_ou_peso_nulos"] = len(df)
    df = df[df["ano"].isin(cfg.anos)]
    contagem["7_data_de_atracacao_em_2022_2024"] = len(df)

    treino = df[df["ano"].isin(cfg.anos_treino)].sort_values("data_atracacao").reset_index(drop=True)
    teste = df[df["ano"] == cfg.ano_teste].sort_values("data_atracacao").reset_index(drop=True)
    contagem["8a_treino_antes_do_corte"] = len(treino)
    contagem["8b_teste_antes_do_corte"] = len(teste)

    ta_max_h = float(treino["ta"].quantile(cfg.ta_quantil_max)) if cfg.ta_quantil_max else None
    if ta_max_h:
        treino = treino[treino["ta"] <= ta_max_h].reset_index(drop=True)
        if cfg.corte_no_teste:
            teste = teste[teste["ta"] <= ta_max_h].reset_index(drop=True)
    contagem["9a_treino_final"] = len(treino)
    contagem["9b_teste_final"] = len(teste)
    mask_trunc = (teste["ta"] <= ta_max_h).values if ta_max_h else np.ones(len(teste), bool)
    return treino, teste, ta_max_h, mask_trunc


def tabela_funil(contagem):
    anterior = {"2_com_registro_de_tempo": "1_atracacoes_brutas_2022_2024",
                "3_com_registro_de_carga": "2_com_registro_de_tempo",
                "4_longo_curso_e_cabotagem": "3_com_registro_de_carga",
                "5_sem_incompletos": "4_longo_curso_e_cabotagem",
                "6_sem_TA_ou_peso_nulos": "5_sem_incompletos",
                "7_data_de_atracacao_em_2022_2024": "6_sem_TA_ou_peso_nulos",
                "9a_treino_final": "8a_treino_antes_do_corte",
                "9b_teste_final": "8b_teste_antes_do_corte"}
    funil = pd.Series(contagem, name="registros").to_frame()
    funil["removidos"] = [contagem[anterior[k]] - v if k in anterior else None for k, v in contagem.items()]
    return funil
