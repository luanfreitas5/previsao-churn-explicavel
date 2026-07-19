"""Fábrica de barras de progresso ``rich`` padronizadas para o projeto."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def build_progress() -> Progress:
    """Cria uma barra de progresso ``rich`` com as colunas padrão do projeto.

    Returns
    -------
    rich.progress.Progress
        Objeto de progresso pronto para uso em ``with``.

    Examples
    --------
    >>> with build_progress() as progress:  # doctest: +SKIP
    ...     task = progress.add_task("Processando", total=3)
    ...     for _ in range(3):
    ...         progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
