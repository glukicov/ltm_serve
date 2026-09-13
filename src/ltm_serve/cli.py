"""Command line entry points: `uv run ltm-serve --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def bench(
    names: Annotated[list[str], typer.Argument(help="sweep names, or 'all'")],
    device: Annotated[str, typer.Option(help="cpu | mps | cuda")] = "mps",
    dtype: Annotated[str, typer.Option(help="bf16 | fp32")] = "bf16",
    suffix: Annotated[str, typer.Option(help="appended to the results file name")] = "",
) -> None:
    """Run benchmark sweeps; results land in results/<name><suffix>.parquet."""
    from ltm_serve.bench import run_sweep
    from ltm_serve.sweeps import sweeps

    available = sweeps(device, dtype)
    selected = list(available) if names == ["all"] else names
    for name in selected:
        configs, runner = available[name]
        print(f"== {name}: {len(configs)} configs -> {run_sweep(name + suffix, configs, runner)}")


@app.command()
def export_checkpoint(
    out_dir: Annotated[Path, typer.Argument(help="directory for model.safetensors + config.json")],
    dtype: Annotated[str, typer.Option()] = "bf16",
) -> None:
    """Write a pre-cast checkpoint for fast, low-memory cold starts."""
    from ltm_serve.model import export_checkpoint as export

    print(export(out_dir, dtype))


@app.command()
def plots() -> None:
    """Render figures from results/*.parquet into docs/figures/."""
    from ltm_serve.plots import render_all

    for path in render_all():
        print(path)
