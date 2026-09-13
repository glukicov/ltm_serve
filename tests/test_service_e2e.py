"""End to end through the HTTP service with the real model: register a context over JSON, predict over both APIs,
and check the answers against stock TabFM. Loads the fp32 checkpoint on CPU (slow)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tabfm import TabFMClassifier

from ltm_serve.data import make_task, penguins_task
from ltm_serve.model import load_model
from ltm_serve.service.app import Settings, create_app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    settings = Settings(
        device="cpu",
        dtype="fp32",
        demo_context="demo:n_train=64,n_features=10,n_classes=2,n_estimators=2",
        max_batch_rows=64,
    )
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.mark.slow
def test_health_and_demo_context(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["contexts"] == 1
    assert client.get("/v2/models/demo/ready").json() == {"ready": True}
    assert client.get("/v2/models/nope/ready").status_code == 404


@pytest.mark.slow
def test_json_context_matches_stock_tabfm(client: TestClient) -> None:
    """Penguins has string categoricals and missing values: the JSON path must preserve them."""
    task = penguins_task()
    rows = task.X_train.astype(object).where(task.X_train.notna(), None).to_numpy().tolist()
    created = client.post(
        "/v1/contexts",
        json={
            "columns": list(task.X_train.columns),
            "rows": rows,
            "labels": task.y_train.tolist(),
            "n_estimators": 3,
            "context_id": "penguins",
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["n_train"] == len(task.X_train) and body["classes"] == ["FEMALE", "MALE"]

    query = task.X_test.iloc[:12]
    query_rows = query.astype(object).where(query.notna(), None).to_numpy().tolist()
    predicted = client.post("/v1/contexts/penguins/predict", json={"rows": query_rows})
    assert predicted.status_code == 200, predicted.text
    got = np.asarray(predicted.json()["probabilities"])

    stock = TabFMClassifier(load_model("cpu", "fp32").model, n_estimators=3, random_state=0)
    expected = stock.fit(task.X_train, task.y_train).predict_proba(query)
    np.testing.assert_allclose(got, expected, atol=1e-4)
    assert predicted.json()["labels"] == [["FEMALE", "MALE"][i] for i in expected.argmax(axis=1)]


@pytest.mark.slow
def test_v2_infer_and_metrics(client: TestClient) -> None:
    rows = make_task(64, 3, 10, 2).X_test.to_numpy(dtype=np.float32)
    payload = {
        "inputs": [{"name": "rows", "shape": list(rows.shape), "datatype": "FP32", "data": rows.ravel().tolist()}]
    }
    resp = client.post("/v2/models/demo/infer", json=payload)
    assert resp.status_code == 200, resp.text
    out = resp.json()["outputs"][0]
    assert out["shape"] == [3, 2]
    probs = np.asarray(out["data"]).reshape(3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    metrics = client.get("/metrics").text
    assert "ltm_batch_rows_count" in metrics and "ltm_request_seconds_count" in metrics


@pytest.mark.slow
def test_unknown_context_is_404(client: TestClient) -> None:
    assert client.post("/v1/contexts/missing/predict", json={"rows": [[0.0] * 10]}).status_code == 404
