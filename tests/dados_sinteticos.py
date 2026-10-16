"""Gera arquivos no layout do Estatístico Aquaviário da ANTAQ para testar o pipeline sem os dados reais."""
import os
import zipfile

import numpy as np
import pandas as pd

NATUREZAS = ["Granel Sólido", "Granel Líquido e Gasoso", "Carga Conteinerizada", "Carga Geral"]
NAVEGACOES = ["Longo Curso", "Cabotagem", "Interior"]
PAISES = ["Brasil", "Dinamarca", "Suíça", "França", "Alemanha"]


def _br(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def gerar(pasta, anos=(2022, 2023, 2024), n_por_ano=900, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(pasta, exist_ok=True)
    portos = [f"Porto {c}" for c in "ABCDEF"]
    bercos = [(p, f"Berço {i}") for p in portos for i in range(1, 4)]
    efeito_berco = {b: rng.uniform(0.6, 1.6) for b in bercos}
    id_atr, id_carga = 1000, 1
    for ano in anos:
        atr, tempos, carga = [], [], []
        for _ in range(n_por_ano):
            id_atr += 1
            porto, berco = bercos[rng.integers(len(bercos))]
            nat = NATUREZAS[rng.integers(len(NATUREZAS))]
            inicio = pd.Timestamp(f"{ano}-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 525_000)))
            peso = float(rng.lognormal(9, 1))
            ta = max(0.5, (6 + 2.5 * np.log1p(peso)) * efeito_berco[(porto, berco)] * rng.lognormal(0, 0.3))
            fim = inicio + pd.Timedelta(hours=ta)
            atr.append({"IDAtracacao": id_atr, "Berço": berco, "Porto Atracação": porto,
                        "Data Atracação": inicio.strftime("%d/%m/%Y %H:%M:%S"),
                        "Data Desatracação": fim.strftime("%d/%m/%Y %H:%M:%S"),
                        "Tipo de Navegação da Atracação": NAVEGACOES[rng.choice(3, p=[.6, .35, .05])],
                        "Nacionalidade do Armador": PAISES[rng.integers(len(PAISES))],
                        "Tipo da Autoridade Portuária": rng.choice(["Porto Organizado", "TUP"])})
            tempos.append({"IDAtracacao": id_atr, "TEsperaAtracacao": _br(1.0), "TAtracado": _br(ta, 2)})
            for parte in rng.dirichlet(np.ones(rng.integers(1, 4))):
                id_carga += 1
                teu = int(parte * peso / 12) if nat == "Carga Conteinerizada" else 0
                carga.append({"IDCarga": id_carga, "IDAtracacao": id_atr, "Natureza da Carga": nat,
                              "VLPesoCargaBruta": _br(parte * peso), "TEU": teu,
                              "FlagMCOperacaoCarga": 1, "ConteinerEstado": "Cheio" if teu else ""})
        arquivos = {f"{ano}Atracacao.txt": atr, f"{ano}TemposAtracacao.txt": tempos, f"{ano}Carga.txt": carga}
        # Como a ANTAQ distribui: um zip por arquivo, texto em Latin-1 separado por ';'
        for nome, linhas in arquivos.items():
            texto = pd.DataFrame(linhas).to_csv(sep=";", index=False)
            with zipfile.ZipFile(os.path.join(pasta, nome.replace(".txt", ".zip")), "w") as zf:
                zf.writestr(nome, texto.encode("latin-1"))
    return pasta


if __name__ == "__main__":
    print(gerar("dados_sinteticos"))
