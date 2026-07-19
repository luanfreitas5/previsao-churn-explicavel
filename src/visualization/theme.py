"""Paleta de cores e tema visual compartilhados do projeto."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns

# Paleta consistente reutilizada em todo o projeto.
CHURN_PALETTE: Final[dict[str, str]] = {
    "churn": "#d62728",  # vermelho — risco de não recompra
    "retencao": "#2ca02c",  # verde — recompra
    "neutro": "#1f77b4",  # azul — elementos neutros
    "destaque": "#ff7f0e",  # laranja — destaques
}


def apply_theme() -> None:
    """Aplica o tema visual padrão (seaborn + matplotlib).

    Define estilo, contexto e DPI para figuras de qualidade de publicação.

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"
