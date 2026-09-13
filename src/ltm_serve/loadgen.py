"""Open-loop load generator for any server speaking the native API or Open Inference Protocol v2.

Open loop means requests are *scheduled* on a Poisson clock at the offered rate regardless of how fast the server
answers — the way real users arrive. A closed-loop tool (N workers each waiting for its previous reply) slows down
exactly when the server does, hiding queueing delay ("coordinated omission"). Latency here is measured from each
request's scheduled send time, so a generator that falls behind is charged to the result, not silently absorbed.

    uv run python -m ltm_serve.loadgen --url http://localhost:8080 --model demo --rate 5 --duration 30
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import httpx
import numpy as np
import typer


@dataclass(frozen=True)
class LoadSpec:
    url: str
    model: str
    protocol: str  # "v1" (native) | "v2" (Triton / KServe / native v2 endpoint)
    input_name: str
    n_features: int
    rows_per_request: int
    rate: float  # offered requests per second
    duration_s: float
    max_inflight: int
    seed: int = 0


@dataclass
class LoadResult:
    spec: LoadSpec
    sent: int
    ok: int
    errors: int
    dropped: int  # not sent because max_inflight was reached: the server is saturated
    achieved_rps: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_server_ms: float

    def row(self) -> dict[str, Any]:
        return asdict(self.spec) | {k: v for k, v in asdict(self).items() if k != "spec"}


def _payloads(spec: LoadSpec, rng: np.random.Generator, n: int = 256) -> tuple[str, list[bytes]]:
    """Pre-serialised request bodies: JSON encoding per request would make the *client* the bottleneck."""
    path = f"/v1/contexts/{spec.model}/predict" if spec.protocol == "v1" else f"/v2/models/{spec.model}/infer"
    bodies = []
    for _ in range(n):
        rows = rng.normal(size=(spec.rows_per_request, spec.n_features)).astype(np.float32)
        if spec.protocol == "v1":
            body: dict[str, Any] = {"rows": rows.tolist()}
        else:
            tensor = {
                "name": spec.input_name,
                "shape": list(rows.shape),
                "datatype": "FP32",
                "data": rows.ravel().tolist(),
            }
            body = {"inputs": [tensor]}
        bodies.append(json.dumps(body).encode())
    return path, bodies


async def _run_shard(spec: LoadSpec) -> dict[str, Any]:
    """One process's share of the load: an independent Poisson stream at `spec.rate`."""
    rng = np.random.default_rng(spec.seed)
    path, bodies = _payloads(spec, rng)
    latencies: list[float] = []
    server_ms: list[float] = []
    errors = dropped = 0
    inflight = 0
    limits = httpx.Limits(max_connections=spec.max_inflight, max_keepalive_connections=spec.max_inflight)
    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(base_url=spec.url, timeout=120.0, limits=limits) as client:

        async def one(scheduled: float, body: bytes) -> None:
            nonlocal errors, inflight
            try:
                resp = await client.post(path, content=body, headers=headers)
                done = time.perf_counter()
                if resp.status_code != 200:
                    errors += 1
                    return
                latencies.append(done - scheduled)
                if spec.protocol == "v1":
                    server_ms.append(float(resp.json()["timing"]["server_ms"]))
            except httpx.HTTPError:
                errors += 1
            finally:
                inflight -= 1

        tasks: list[asyncio.Task[None]] = []
        start = time.perf_counter()
        next_at = start
        i = 0
        while next_at - start < spec.duration_s:
            now = time.perf_counter()
            if next_at > now:
                await asyncio.sleep(next_at - now)
            if inflight >= spec.max_inflight:
                dropped += 1
            else:
                inflight += 1
                tasks.append(asyncio.create_task(one(next_at, bodies[i % len(bodies)])))
                i += 1
            next_at += rng.exponential(1.0 / spec.rate)
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
    return {
        "latencies": latencies,
        "server_ms": server_ms,
        "sent": len(tasks),
        "errors": errors,
        "dropped": dropped,
        "elapsed": elapsed,
    }


def _shard_entry(spec: LoadSpec) -> dict[str, Any]:
    return asyncio.run(_run_shard(spec))


def run_load(spec: LoadSpec, procs: int = 1) -> LoadResult:
    """Split the offered rate across `procs` processes; merged Poisson streams are still Poisson."""
    shards = [
        replace(spec, rate=spec.rate / procs, max_inflight=max(1, spec.max_inflight // procs), seed=spec.seed + k)
        for k in range(procs)
    ]
    if procs == 1:
        parts = [_shard_entry(shards[0])]
    else:
        with ProcessPoolExecutor(max_workers=procs, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_shard_entry, shards))
    latencies = [x for part in parts for x in part["latencies"]]
    server_ms = [x for part in parts for x in part["server_ms"]]
    elapsed = max(part["elapsed"] for part in parts)
    lat = np.asarray(latencies) * 1e3 if latencies else np.asarray([float("nan")])
    return LoadResult(
        spec=spec,
        sent=sum(part["sent"] for part in parts),
        ok=len(latencies),
        errors=sum(part["errors"] for part in parts),
        dropped=sum(part["dropped"] for part in parts),
        achieved_rps=len(latencies) / elapsed,
        p50_ms=float(np.percentile(lat, 50)),
        p90_ms=float(np.percentile(lat, 90)),
        p95_ms=float(np.percentile(lat, 95)),
        p99_ms=float(np.percentile(lat, 99)),
        max_ms=float(lat.max()),
        mean_server_ms=float(np.mean(server_ms)) if server_ms else float("nan"),
    )


def main(
    url: Annotated[str, typer.Option()] = "http://localhost:8080",
    model: Annotated[str, typer.Option(help="context id (native) or model name (v2)")] = "demo",
    protocol: Annotated[str, typer.Option(help="v1 | v2")] = "v2",
    input_name: Annotated[str, typer.Option(help="v2 input tensor name")] = "rows",
    n_features: Annotated[int, typer.Option()] = 10,
    rows: Annotated[int, typer.Option(help="rows per request")] = 1,
    rates: Annotated[str, typer.Option(help="comma-separated offered rates (req/s) to sweep")] = "1",
    duration: Annotated[float, typer.Option(help="seconds per rate")] = 30.0,
    max_inflight: Annotated[int, typer.Option()] = 256,
    warmup: Annotated[float, typer.Option(help="seconds of load at the first rate before measuring")] = 5.0,
    out: Annotated[Path | None, typer.Option(help="append results as JSON lines")] = None,
    label: Annotated[str, typer.Option(help="free-form tag stored with each result")] = "",
    procs: Annotated[int, typer.Option(help="client processes; raise it until the client is not the bottleneck")] = 1,
) -> None:
    """Sweep offered load and report latency percentiles per rate."""
    rate_list = [float(r) for r in rates.split(",")]
    spec0 = LoadSpec(url, model, protocol, input_name, n_features, rows, rate_list[0], warmup, max_inflight)
    if warmup > 0:
        run_load(spec0, procs)
    for rate in rate_list:
        result = run_load(
            LoadSpec(url, model, protocol, input_name, n_features, rows, rate, duration, max_inflight), procs
        )
        row = result.row() | {"label": label, "procs": procs}
        print(
            f"rate={rate:>7.2f} achieved={result.achieved_rps:7.2f} ok={result.ok} err={result.errors} "
            f"drop={result.dropped} p50={result.p50_ms:8.1f} p95={result.p95_ms:8.1f} p99={result.p99_ms:8.1f} ms",
            flush=True,
        )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    typer.run(main)
