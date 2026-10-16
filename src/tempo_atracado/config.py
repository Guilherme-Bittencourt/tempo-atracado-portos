"""Configuração do experimento.

Os valores padrão reproduzem o experimento descrito no trabalho.
"""
from dataclasses import dataclass, field
import os

# Atributos usados pelos modelos (estruturais: mudar aqui invalida os modelos em cache)
CAT_COLS = ["porto", "berco_id", "tipo_navegacao", "natureza", "armador"]
TIME_COLS = ["mes"]
NUM_COLS = ["log_peso", "log_teu", "log_registros_carga"]

# Ordem de treino: o mais importante primeiro
MODELOS_PADRAO = ["CatBoost", "Random Forest", "XGBoost", "Ridge"]


@dataclass
class Config:
    # Pastas
    dir_dados: str = "dados"            # onde estão os .zip/.txt da ANTAQ (busca recursiva)
    dir_saida: str = "resultados"       # tabelas, figuras, modelos e resumo

    # Período e recorte
    anos_treino: list = field(default_factory=lambda: [2022, 2023])
    ano_teste: int = 2024
    navegacoes: list = field(default_factory=lambda: ["Longo Curso", "Cabotagem"])
    origem_ta: str = "datas"               # "datas" = Desatracação − Atracação (como no artigo);
                                           # "antaq" = coluna TAtracado do arquivo TemposAtracacao
    ta_quantil_max: float | None = 0.995   # corte de extremos, calculado só no treino
    corte_no_teste: bool = False           # True = comportamento da versão original

    # Modelagem
    modelos: list = field(default_factory=lambda: list(MODELOS_PADRAO))
    n_iter: int = 15
    n_splits_cv: int = 3
    amostra_tuning: int | None = None
    k_min: int = 2
    k_max: int = 10
    k_fixo: int | None = None              # None = k de maior silhueta
    n_bootstrap: int = 1000
    n_shap: int = 2000
    seed: int = 42

    # Experimentos complementares
    rodar_ablacao: bool = True
    rodar_xgb_nativo: bool = False         # opcional, ~10–20 min
    rodar_sementes: bool = False           # opcional, ~10–30 min
    sementes_extras: list = field(default_factory=lambda: [1, 7, 123, 2024])

    # Exibição (True no notebook, False na linha de comando)
    mostrar_figuras: bool = False

    @property
    def anos(self):
        return list(self.anos_treino) + [self.ano_teste]

    @property
    def k_range(self):
        return range(self.k_min, self.k_max + 1)

    @property
    def dir_modelos(self):
        return os.path.join(self.dir_saida, "modelos")

    @property
    def dir_extraidos(self):
        return os.path.join(self.dir_dados, "extraidos")
