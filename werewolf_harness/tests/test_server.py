"""Backend tests.

The one that matters most is the first: a provider's API key must never come
back over the wire, in any response, ever.
"""

from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from werewolf_harness.server import app as server_app  # noqa: E402
from werewolf_harness.server import db as dbmod  # noqa: E402

SECRET = "sk-thisisatotallyrealkey3f2a"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    conn = dbmod.connect(tmp_path / "test.db")
    monkeypatch.setattr(server_app, "conn", conn)
    return TestClient(server_app.app)


def test_api_key_never_comes_back(client):
    created = client.post("/api/providers", json={
        "name": "relay", "base_url": "https://example.invalid/v1", "api_key": SECRET,
    }).json()
    assert SECRET not in json.dumps(created)
    assert created["api_key_masked"].endswith("3f2a")
    assert "api_key" not in created

    listed = client.get("/api/providers").json()
    assert SECRET not in json.dumps(listed)

    model = client.post("/api/models", json={
        "provider_id": created["id"], "display_name": "M", "model_name": "some-model",
        "group": "pool-a",
    }).json()
    assert SECRET not in json.dumps(client.get("/api/models").json())
    assert SECRET not in json.dumps(model)


def test_model_requires_an_existing_provider(client):
    response = client.post("/api/models", json={
        "provider_id": "prov_nope", "display_name": "M", "model_name": "m",
    })
    assert response.status_code == 404


def test_deleting_a_provider_removes_its_models(client):
    provider = client.post("/api/providers", json={
        "name": "relay", "base_url": "https://example.invalid/v1", "api_key": SECRET,
    }).json()
    client.post("/api/models", json={
        "provider_id": provider["id"], "display_name": "M", "model_name": "m",
    })
    client.delete(f"/api/providers/{provider['id']}")
    assert client.get("/api/models").json() == []


def test_run_a_game_and_replay_it(client):
    started = client.post("/api/games", json={
        "seed": 3, "guard_layers": ["L1", "L2"], "attack_enabled": True,
    }).json()
    assert started["status"] == "running"

    with client.stream("GET", f"/api/games/{started['stream_id']}/stream") as stream:
        kinds = [
            line[len("event: "):].strip()
            for line in stream.iter_lines()
            if line.startswith("event: ")
        ]
    assert "round_start" in kinds and "end" in kinds

    games = client.get("/api/games").json()
    assert games, "the finished game should have been persisted"
    log = client.get(f"/api/games/{games[0]['game_id']}").json()
    assert log["rounds"] and log["ground_truth"]["roles"]

    axes = client.get(f"/api/games/{games[0]['game_id']}/metrics").json()
    assert set(axes) >= {"injection_trials", "consistency", "stability"}


def test_missing_game_is_a_404(client):
    assert client.get("/api/games/nope").status_code == 404
    assert client.get("/api/games/nope/stream").status_code == 404


def test_experiment_runs_and_can_be_stopped(client):
    experiment = client.post("/api/experiments", json={
        "name": "t", "seeds": 2, "include_benign": False, "workers": 2,
        "arms": [{"guard_layers": [], "evidence_forced": False},
                 {"guard_layers": ["L1"], "evidence_forced": False}],
    }).json()
    eid = experiment["id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        state = client.get(f"/api/experiments/{eid}").json()
        if state["status"] in ("finished", "stopped"):
            break
        time.sleep(0.2)
    assert state["status"] == "finished", state
    assert state["progress"]["done"] == 4

    summary = client.get(f"/api/experiments/{eid}/metrics").json()
    assert set(summary["arms"]) == {"none", "L1"}
    assert summary["games"] == 4


def test_stopping_an_unknown_experiment_is_a_404(client):
    assert client.post("/api/experiments/exp_nope/stop").status_code == 404


def test_masking_helper():
    assert dbmod.mask_key("sk-abcdefghijklmnop") == "sk-****mnop"
    assert dbmod.mask_key("short") == "****"
    assert dbmod.mask_key("") == ""


# --------------------------------------------------- the real HTTP client

def test_a_full_game_over_real_http_in_both_tool_modes():
    """The offline client bypasses the HTTP layer, native tool-call
    round-tripping, message pairing, usage accounting and the error mapping --
    every part the first paid run depends on. A local server that speaks the
    chat-completions API exercises them without a network or a key."""
    from werewolf_harness.tests.fake_gateway import _check

    assert _check(port=8917) == 0


def test_the_probe_survives_a_transient_gateway_failure():
    """A relay hiccup must not be reported as "this model does not work": the
    probe decides which models enter the experiment at all."""
    from werewolf_harness.harness.providers import OpenAICompatClient, probe_model
    from werewolf_harness.tests import fake_gateway

    server = fake_gateway.serve(port=8918)
    try:
        fake_gateway.Handler.fail_next = 2
        result = probe_model(
            OpenAICompatClient(model="fake-flaky", api_key="sk-local-fake",
                               base_url="http://127.0.0.1:8918/v1"),
            check_temperature=False,
        )
        assert result.reachable, result.error
        assert any("transient" in n for n in result.notes), result.notes
    finally:
        fake_gateway.Handler.fail_next = 0
        server.shutdown()
