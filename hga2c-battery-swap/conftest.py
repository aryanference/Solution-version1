"""Pytest configuration and shared fixtures for HGA²C test suite."""
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `from env.masks import ...` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures: load configs once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def instance_data(project_root: Path) -> dict[str, Any]:
    """Load the target instance from configs/instance.json."""
    path = project_root / "configs" / "instance.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def economics_config(project_root: Path) -> dict[str, Any]:
    """Load economic parameters from configs/economics.yaml."""
    path = project_root / "configs" / "economics.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def hyperparams(project_root: Path) -> dict[str, Any]:
    """Load training hyperparameters from configs/hyperparams.yaml."""
    path = project_root / "configs" / "hyperparams.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def hub_coords(instance_data: dict) -> tuple[float, float]:
    """Hub (x, y) coordinates."""
    return (instance_data["hub"]["x"], instance_data["hub"]["y"])


@pytest.fixture(scope="session")
def region_coords(instance_data: dict) -> list[tuple[float, float]]:
    """List of (x, y) for each region, indexed by region id."""
    return [(r["x"], r["y"]) for r in instance_data["regions"]]


@pytest.fixture(scope="session")
def all_node_coords(hub_coords, region_coords) -> list[tuple[float, float]]:
    """All node coords: index 0 = Hub, index 1..9 = Region 0..8."""
    return [hub_coords] + region_coords


@pytest.fixture(scope="session")
def demand(instance_data: dict) -> list[int]:
    """Demand vector D_r for each region."""
    return [r["demand"] for r in instance_data["regions"]]


@pytest.fixture(scope="session")
def s_hat(instance_data: dict) -> list[int]:
    """Functional scooter counts Ŝ_r per region."""
    return [r["functional"] for r in instance_data["regions"]]


@pytest.fixture(scope="session")
def s_check(instance_data: dict) -> list[int]:
    """Non-functional scooter counts S̆_r per region."""
    return [r["non_functional"] for r in instance_data["regions"]]
