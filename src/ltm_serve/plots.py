"""Figures for docs/WRITEUP.md, rendered from results/*.parquet (Apple M4), results/cuda/*_cuda.parquet (NVIDIA L4)
and results/load/*.jsonl (load tests on GKE).

Encoding is fixed across every figure so it reads as one system: colour = what is being compared (stock = orange,
cached = blue, a third series = aqua; the first three validated slots of the dataviz reference palette), line style =
hardware (solid = L4, dashed = M4). Every series is also named in the legend, so identity never rests on colour alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from ltm_serve.bench import RESULTS_DIR

FIGURES = RESULTS_DIR.parent / "docs" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODE_COLOUR = {"stock": ORANGE, "cached": BLUE}
PLATFORM_STYLE = {"L4": "-", "M4": "--"}
INK, INK_2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c2"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "font.size": 10,
    }
)


def _plain_log_ticks(fig: Figure) -> None:
    """Label log axes with plain numbers (1, 2, 5, 10, 20, ...) instead of overlapping 3x10^1-style labels."""
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "log":
                axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                axis.set_major_formatter(fmt)
                axis.set_minor_formatter(NullFormatter())


def _save(fig: Figure, name: str, rect: tuple[float, float, float, float] = (0, 0, 1, 1)) -> Path:
    _plain_log_ticks(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _load(name: str) -> pd.DataFrame | None:
    """Both platforms' results for a sweep, with a `platform` column (L4 = CUDA, M4 = Apple MPS/CPU)."""
    frames = []
    sources = ((RESULTS_DIR / "cuda" / f"{name}_cuda.parquet", "L4"), (RESULTS_DIR / f"{name}.parquet", "M4"))
    for path, platform in sources:
        if path.exists():
            frames.append(pd.read_parquet(path).assign(platform=platform))
    return pd.concat(frames, ignore_index=True) if frames else None


def _slope(x: Iterable[float], y: Iterable[float]) -> float:
    """Log-log slope: 1 = linear, 2 = quadratic, 0 = flat."""
    return float(np.polyfit(np.log(list(x)), np.log(list(y)), 1)[0])


def _lines(ax: Axes, df: pd.DataFrame, x: str, y: str, slope_from: float | None = None) -> None:
    """One line per (platform, mode). `slope_from` fits the log-log slope over x >= that value."""
    for (platform, mode), part in df.groupby(["platform", "mode"], sort=False):
        part = part.sort_values(x)
        label = f"{platform} {mode}"
        if slope_from is not None:
            fit = part[part[x] >= slope_from]
            if len(fit) > 2:
                label += f" (slope {_slope(fit[x], fit[y]):.2f})"
        style = PLATFORM_STYLE[str(platform)]
        ax.plot(part[x], part[y], marker="o", color=MODE_COLOUR[str(mode)], linestyle=style, label=label)


def context_scaling() -> Path | None:
    df = _load("context_scaling")
    if df is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    _lines(ax1, df, "n_train", "p50_ms", slope_from=512)
    ax1.set(xscale="log", yscale="log", xlabel="context (training) rows", ylabel="p50 latency, 16-row request (ms)")
    ax1.set_title("Latency vs context size (4 ensemble members)")
    ax1.legend(fontsize=8)
    cached = df[df["mode"] == "cached"]
    for platform, part in cached.groupby("platform"):
        part = part.sort_values("n_train")
        style = PLATFORM_STYLE[str(platform)]
        ax2.plot(
            part["n_train"], part["encode_s"], marker="o", color=BLUE, linestyle=style, label=f"{platform} encode (s)"
        )
    l4 = cached[cached["platform"] == "L4"].sort_values("n_train")
    if not l4.empty:
        ax2.plot(l4["n_train"], l4["cache_mb"] / 1e3, marker="s", color=AQUA, label="cache size, bf16 (GB)")
    ax2.set(xscale="log", yscale="log", xlabel="context (training) rows")
    ax2.set_title("What the cache costs (one-off, per context)")
    ax2.legend(fontsize=8)
    return _save(fig, "context_scaling")


def feature_scaling() -> Path | None:
    df = _load("feature_scaling")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    _lines(ax, df, "n_features", "p50_ms", slope_from=16)
    ax.set(xscale="log", yscale="log", xlabel="features (columns)", ylabel="p50 latency, 16-row request (ms)")
    ax.set_title("Latency vs table width (256 context rows)")
    ax.legend(fontsize=8)
    return _save(fig, "feature_scaling")


def query_scaling() -> Path | None:
    df = _load("query_scaling")
    if df is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    _lines(ax1, df, "n_query", "p50_ms")
    ax1.set(xscale="log", yscale="log", xlabel="rows per request", ylabel="p50 latency (ms)")
    ax1.set_title("Latency vs request size (512 context rows)")
    ax1.legend(fontsize=8)
    _lines(ax2, df, "n_query", "rows_per_s")
    ax2.set(xscale="log", yscale="log", xlabel="rows per request", ylabel="rows / s")
    ax2.set_title("Throughput vs request size")
    ax2.legend(fontsize=8)
    return _save(fig, "query_scaling")


def ensemble() -> Path | None:
    df = _load("ensemble")
    if df is None:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for platform, part in df.groupby("platform"):
        part = part.sort_values("n_estimators")
        style = PLATFORM_STYLE[str(platform)]
        ax1.plot(part["n_estimators"], part["p50_ms"], marker="o", color=BLUE, linestyle=style, label=f"{platform}")
    ax1.set(xscale="log", yscale="log", xlabel="ensemble members", ylabel="p50 latency, 1-row request (ms)")
    ax1.set_title("Ensemble size: latency")
    ax1.legend(fontsize=8)
    for platform, part in df.groupby("platform"):
        part = part.sort_values("n_estimators")
        style = PLATFORM_STYLE[str(platform)]
        ax2.plot(part["n_estimators"], part["accuracy"], marker="o", color=BLUE, linestyle=style, label=f"{platform}")
        if "log_loss" in part and part["log_loss"].notna().any():
            for _, r in part.iterrows():
                ax2.annotate(
                    f"LL {r['log_loss']:.3f}",
                    (r["n_estimators"], r["accuracy"]),
                    textcoords="offset points",
                    xytext=(0, -14),
                    fontsize=7,
                    color=INK_2,
                    ha="center",
                )
    ax2.set(xscale="log", xlabel="ensemble members", ylabel="accuracy (1,024 held-out rows)")
    ax2.set_ylim(df["accuracy"].min() - 0.03, df["accuracy"].max() + 0.03)
    ax2.set_title("Ensemble size: quality (LL = log loss, M4 run)")
    ax2.legend(fontsize=8)
    return _save(fig, "ensemble")


def member_batching() -> Path | None:
    df = _load("member_batching")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for (platform, n_query), part in df.groupby(["platform", "n_query"]):
        part = part.sort_values("member_batch_size")
        colour = BLUE if n_query == 1 else AQUA
        style = PLATFORM_STYLE[str(platform)]
        label = f"{platform}, {n_query}-row request"
        ax.plot(part["member_batch_size"], part["p50_ms"], marker="o", color=colour, linestyle=style, label=label)
    ax.set(xscale="log", yscale="log", xlabel="ensemble members per forward pass (of 16)", ylabel="p50 latency (ms)")
    ax.set_title("Batching ensemble members")
    ax.legend(fontsize=8)
    return _save(fig, "member_batching")


def stages() -> Path | None:
    df = _load("stages")
    if df is None:
        return None
    parts = {
        "embedding + column/row stages": ["cell_embedder_s", "col_embedder_s", "row_interactor_s", "col_embedder_2_s"],
        "row stage 2": ["row_interactor_2_s"],
        "in-context encoder (24 layers)": ["icl_predictor_s"],
    }
    platforms = [p for p in ("L4", "M4") if p in set(df["platform"])]
    fig, axes = plt.subplots(1, len(platforms), figsize=(6 * len(platforms), 4.8), squeeze=False)
    for ax, platform in zip(axes[0], platforms, strict=True):
        part = df[df["platform"] == platform].reset_index(drop=True)
        labels = [f"{r.n_train} rows\n{r.n_features} feat" for r in part.itertuples()]
        bottom = np.zeros(len(part))
        for colour, (name, cols) in zip((AQUA, ORANGE, BLUE), parts.items(), strict=True):
            share = part[cols].sum(axis=1).to_numpy() / part["total_s"].to_numpy()
            ax.bar(labels, share, bottom=bottom, color=colour, label=name, edgecolor=SURFACE, linewidth=2)
            bottom += share
        rest = 1 - bottom
        ax.bar(labels, rest, bottom=bottom, color=NEUTRAL, label="outside the model", edgecolor=SURFACE, linewidth=2)
        for i, total in enumerate(part["total_s"]):
            ax.annotate(f"{total:.1f}s", (i, 1.0), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="share of stock predict time", ylim=(0, 1.1))
        ax.set_title(f"Where stock inference time goes ({platform})")
        ax.tick_params(axis="x", labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    return _save(fig, "stages", rect=(0, 0.08, 1, 1))


def overhead() -> Path | None:
    df = _load("overhead")
    if df is None:
        return None
    platforms = [p for p in ("L4", "M4") if p in set(df["platform"])]
    fig, axes = plt.subplots(1, len(platforms), figsize=(6 * len(platforms), 4.4), squeeze=False)
    for ax, platform in zip(axes[0], platforms, strict=True):
        part = df[df["platform"] == platform].sort_values(["n_query", "n_estimators"]).reset_index(drop=True)
        labels = [f"{r.n_query} rows\n{r.n_estimators} est" for r in part.itertuples()]
        ax.bar(labels, part["preprocess_ms"], color=AQUA, label="host preprocessing", edgecolor=SURFACE, linewidth=2)
        ax.bar(
            labels,
            part["forward_ms"],
            bottom=part["preprocess_ms"],
            color=BLUE,
            label="forward pass",
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="median ms per request")
        ax.set_title(f"Cached request: preprocessing vs forward ({platform})")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=8)
    return _save(fig, "overhead")


def compile_modes() -> Path | None:
    df = _load("compile")
    if df is None:
        return None
    if "L4" in set(df["platform"]):
        df = df[df["platform"] == "L4"]
    df = df.assign(compile_mode=df["compile_mode"].fillna("eager"))
    order = [m for m in ("eager", "default", "reduce-overhead") if m in set(df["compile_mode"])]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    width = 0.8 / len(order)
    queries = sorted(df["n_query"].unique())
    for i, (colour, mode) in enumerate(zip((ORANGE, BLUE, AQUA), order, strict=False)):
        part = df[df["compile_mode"] == mode].set_index("n_query").reindex(queries)
        xs = np.arange(len(queries)) + (i - (len(order) - 1) / 2) * width
        ax.bar(xs, part["p50_ms"], width=width * 0.92, color=colour, label=mode)
        for x, v in zip(xs, part["p50_ms"], strict=True):
            ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(queries)), [f"{q}-row request" for q in queries])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="p50 latency (ms)")
    ax.set_title("torch.compile on the cached forward (L4)")
    ax.legend(fontsize=8)
    return _save(fig, "compile")


def precision() -> Path | None:
    path = RESULTS_DIR / "precision.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.assign(target=df["device"].str.upper() + " " + df["dtype"])
    targets = ["CPU fp32", "CPU bf16", "MPS fp32", "MPS bf16"]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for i, (mode, colour) in enumerate(MODE_COLOUR.items()):
        part = df[df["mode"] == mode].set_index("target").reindex(targets)
        xs = np.arange(len(targets)) + (i - 0.5) * 0.4
        ax.bar(xs, part["p50_ms"], width=0.37, color=colour, label=mode)
        for x, v in zip(xs, part["p50_ms"], strict=True):
            ax.annotate(f"{v:,.0f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(targets)), targets)
    ax.grid(axis="x", visible=False)
    ax.set(yscale="log", ylabel="p50 latency, 16-row request (ms)")
    ax.set_title("Device and precision on Apple M4 (128 context rows)")
    ax.legend(fontsize=8)
    return _save(fig, "precision")


def load_tests() -> list[Path]:
    """One figure per results/load/<name>.jsonl: latency (p50 solid, p99 dotted) and throughput vs offered load."""
    out = []
    for path in sorted((RESULTS_DIR / "load").glob("*.jsonl")):
        df = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for colour, (label, part) in zip((BLUE, ORANGE, AQUA), df.groupby("label", sort=False), strict=False):
            part = part.sort_values("rate")
            ax1.plot(part["rate"], part["p50_ms"], marker="o", color=colour, label=f"{label} p50")
            ax1.plot(part["rate"], part["p99_ms"], marker="^", linestyle=":", color=colour, label=f"{label} p99")
            ax2.plot(part["rate"], part["achieved_rps"], marker="o", color=colour, label=label)
        ax1.set(
            xscale="log", yscale="log", xlabel="offered load (requests/s)", ylabel="latency from scheduled send (ms)"
        )
        ax1.set_title("Latency under load")
        ax1.legend(fontsize=7)
        low, top = float(df["rate"].min()), float(df["rate"].max())
        ax2.plot([low, top], [low, top], color=NEUTRAL, linewidth=1, zorder=0, label="keeping up")
        ax2.set(xscale="log", yscale="log", xlabel="offered load (requests/s)", ylabel="completed requests/s")
        ax2.set_title("Throughput")
        ax2.legend(fontsize=8)
        out.append(_save(fig, f"load_{path.stem}"))
    return out


def render_all() -> list[Path]:
    makers = (
        context_scaling,
        feature_scaling,
        query_scaling,
        ensemble,
        member_batching,
        stages,
        overhead,
        compile_modes,
        precision,
    )
    return [p for p in (f() for f in makers) if p is not None] + load_tests()
