"""Referência ingênua: mediana histórica do TA em cascata (berço × carga → berço → porto → geral)."""
import numpy as np
import pandas as pd

NIVEIS_REF = [("berço × tipo de carga", ["berco_id", "natureza"]), ("berço", ["berco_id"]), ("porto", ["porto"])]


class ReferenciaIngenua:
    def fit(self, treino):
        self.tabelas = {nome: treino.groupby(cols)["ta"].median() for nome, cols in NIVEIS_REF}
        self.global_ = float(treino["ta"].median())
        return self

    def predict(self, d):
        """Retorna (predição em horas, nível da cascata usado em cada linha)."""
        pred = np.full(len(d), np.nan)
        nivel = np.full(len(d), "geral", dtype=object)
        for nome, cols in NIVEIS_REF:
            falta = np.isnan(pred)
            if not falta.any():
                break
            chaves = pd.MultiIndex.from_frame(d.loc[falta, cols]) if len(cols) > 1 else d.loc[falta, cols[0]]
            valores = self.tabelas[nome].reindex(chaves).values.astype(float)
            idx = np.where(falta)[0]
            achou = ~np.isnan(valores)
            pred[idx[achou]] = valores[achou]
            nivel[idx[achou]] = nome
        pred[np.isnan(pred)] = self.global_
        return pred, nivel
