"""Utilitários de saída, exibição e cronometragem."""
import json
import os
import re
import time
import unicodedata

import matplotlib.pyplot as plt


def normalizar(s):
    """Minúsculas, sem acentos e sem nada além de letras e dígitos."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _em_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def mostrar(obj):
    """display() no notebook, print() no terminal."""
    if _em_notebook():
        from IPython.display import display
        display(obj)
    else:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


class Cronometro:
    def __init__(self):
        self.inicio = time.time()

    def minutos(self):
        return (time.time() - self.inicio) / 60

    def etapa(self, titulo):
        print(f"\n{'=' * 90}\n{titulo}   [{self.minutos():.1f} min]\n{'=' * 90}", flush=True)


class Saida:
    """Grava tabelas, figuras e JSON na pasta de resultados."""

    def __init__(self, pasta, mostrar_figuras=False):
        self.pasta = pasta
        self.mostrar_figuras = mostrar_figuras
        os.makedirs(pasta, exist_ok=True)
        plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})

    def caminho(self, nome):
        return os.path.join(self.pasta, nome)

    def csv(self, df, nome, **kw):
        df.to_csv(self.caminho(nome), **kw)

    def fig(self, fig, nome):
        fig.savefig(self.caminho(nome))
        if self.mostrar_figuras:
            plt.show()
        plt.close(fig)

    def json(self, obj, nome):
        with open(self.caminho(nome), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
