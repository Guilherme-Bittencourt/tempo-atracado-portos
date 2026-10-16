"""Linha de comando: python -m tempo_atracado [opções]"""
import argparse

import matplotlib

from .config import MODELOS_PADRAO, Config


def _args(argv=None):
    p = argparse.ArgumentParser(
        prog="tempo-atracado",
        description="Predição do tempo atracado de navios em portos brasileiros (dados da ANTAQ).")
    p.add_argument("--dados", default="dados", help="pasta com os .zip/.txt da ANTAQ (padrão: dados)")
    p.add_argument("--saida", default="resultados", help="pasta de resultados (padrão: resultados)")
    p.add_argument("--modelos", nargs="+", default=MODELOS_PADRAO, choices=MODELOS_PADRAO,
                   help="modelos a treinar (padrão: todos)")
    p.add_argument("--n-iter", type=int, default=15, help="iterações da busca aleatória")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-shap", type=int, default=2000)
    p.add_argument("--k", type=int, default=None, help="fixa o nº de perfis do K-Means (padrão: maior silhueta)")
    p.add_argument("--origem-ta", choices=["datas", "antaq"], default="datas",
                   help="datas = Desatracação − Atracação (padrão, como no artigo); "
                        "antaq = coluna TAtracado do arquivo TemposAtracacao")
    p.add_argument("--corte-no-teste", action="store_true",
                   help="aplica o corte de extremos também ao teste (comportamento da versão original)")
    p.add_argument("--sem-ablacao", action="store_true", help="não roda a ablação dos atributos de volume")
    p.add_argument("--xgb-nativo", action="store_true", help="roda o XGBoost com categóricas nativas")
    p.add_argument("--sementes", action="store_true", help="roda o teste de estabilidade entre sementes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    a = _args(argv)
    matplotlib.use("Agg")
    from .pipeline import executar

    cfg = Config(dir_dados=a.dados, dir_saida=a.saida, modelos=a.modelos, n_iter=a.n_iter,
                 n_bootstrap=a.n_bootstrap, n_shap=a.n_shap, k_fixo=a.k, origem_ta=a.origem_ta,
                 corte_no_teste=a.corte_no_teste,
                 rodar_ablacao=not a.sem_ablacao, rodar_xgb_nativo=a.xgb_nativo, rodar_sementes=a.sementes,
                 seed=a.seed)
    executar(cfg)


if __name__ == "__main__":
    main()
