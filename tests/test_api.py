"""API tests for the pybbn FastAPI backend.

Tests cover: default params, custom params, invalid inputs,
boundary values, and all-params-together scenarios.

NOTE: These tests require the FastAPI backend (api/main.py) to exist.
If the backend hasn't been created yet, tests will be skipped.
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from api.main import app
    HAS_BACKEND = True
except (ImportError, ModuleNotFoundError):
    HAS_BACKEND = False
    app = None

client = TestClient(app) if HAS_BACKEND else None


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_compute_defaults():
    """Test that the endpoint works with default (empty) params."""
    r = client.post("/api/bbn/compute", json={})
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) > 0
    # Check that we get probabilities for each node
    for node in data["nodes"]:
        assert "name" in node
        assert "p_true" in node
        assert "p_false" in node
        assert 0.0 <= node["p_true"] <= 1.0
        assert 0.0 <= node["p_false"] <= 1.0
    # Check SVG is returned
    assert "<svg" in data["assurance_case_svg"]


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_compute_custom_params():
    """Test with custom probability values."""
    r = client.post("/api/bbn/compute", json={
        "p_correct_navigation": 0.5,
        "nav_threshold": 2
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) > 0


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_invalid_probability_too_high():
    """Test that probabilities > 1.0 are rejected."""
    r = client.post("/api/bbn/compute", json={"p_correct_navigation": 1.5})
    assert r.status_code == 422


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_invalid_probability_negative():
    """Test that negative probabilities are rejected."""
    r = client.post("/api/bbn/compute", json={"p_no_collision": -0.1})
    assert r.status_code == 422


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_invalid_threshold_negative():
    """Test that negative thresholds are rejected."""
    r = client.post("/api/bbn/compute", json={"nav_threshold": -1})
    assert r.status_code == 422


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_boundary_values():
    """Test boundary values (0.0 and 1.0 probabilities)."""
    r = client.post("/api/bbn/compute", json={
        "p_correct_navigation": 0.0,
        "p_no_collision": 1.0,
        "p_correct_pose": 0.0
    })
    assert r.status_code == 200


@pytest.mark.skipif(not HAS_BACKEND, reason="FastAPI backend (api.main) not yet available")
def test_all_params_together():
    """Test with all parameters set."""
    r = client.post("/api/bbn/compute", json={
        "p_correct_navigation": 0.95,
        "p_no_collision": 0.05,
        "p_correct_pose": 0.85,
        "nav_threshold": 3,
        "collision_threshold": 1,
        "pose_threshold": 2
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) > 0
    assert "<svg" in data["assurance_case_svg"]
