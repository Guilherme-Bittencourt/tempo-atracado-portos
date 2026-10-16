"""Predição do tempo atracado de navios em portos brasileiros com dados abertos da ANTAQ."""
from .config import Config

__all__ = ["Config", "executar"]
__version__ = "2.0.0"


def executar(cfg=None):
    from .pipeline import executar as _executar
    return _executar(cfg)
