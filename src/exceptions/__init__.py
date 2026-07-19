"""Exceções customizadas do projeto de previsão de churn.

Todas herdam de :class:`ChurnProjectError`, permitindo capturar qualquer erro
de domínio do projeto com um único ``except``.

Classes
-------
ChurnProjectError
    Base de todas as exceções do projeto.
DataValidationError
    Falha de contrato de dados (schema pandera).
DataNotFoundError
    Arquivo de dados esperado ausente.
ModelNotTrainedError
    Operação que exige um modelo treinado/carregado.
ConfigurationError
    Configuração inválida ou incoerente.
"""

from __future__ import annotations


class ChurnProjectError(Exception):
    """Exceção base para todos os erros de domínio do projeto."""


class DataValidationError(ChurnProjectError):
    """Levantada quando um DataFrame viola seu contrato de dados (pandera)."""


class DataNotFoundError(ChurnProjectError):
    """Levantada quando um arquivo de dados esperado não é encontrado."""


class ModelNotTrainedError(ChurnProjectError):
    """Levantada ao usar um modelo que ainda não foi treinado ou carregado."""


class ConfigurationError(ChurnProjectError):
    """Levantada quando a configuração é inválida ou incoerente."""


__all__ = [
    "ChurnProjectError",
    "ConfigurationError",
    "DataNotFoundError",
    "DataValidationError",
    "ModelNotTrainedError",
]
