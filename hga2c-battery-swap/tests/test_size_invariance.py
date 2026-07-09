"""Test to ensure HGA²C policy can handle any instance size (parametric universality)."""
import pytest
import torch
import yaml

from env.battery_swap_env import make_env
from models.hga2c_policy import build_policy_from_config
from data.instance_generator import generate_instance


@pytest.fixture
def policy():
    with open("configs/hyperparams.yaml") as f:
        hp = yaml.safe_load(f)
    return build_policy_from_config(hp)


@pytest.fixture
def economics():
    with open("configs/economics.yaml") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("n_regions", [5, 9, 14])
@pytest.mark.parametrize("n_vehicles", [1, 2, 3])
def test_size_invariance(policy, economics, n_regions, n_vehicles):
    """Test that the policy can perform a forward pass on various sizes."""
    # Generate random instance of given size
    instance = generate_instance(
        n_regions=n_regions,
        n_scooters=n_regions * 2,
        n_vehicles=n_vehicles,
        seed=42 + n_regions + n_vehicles
    )

    env = make_env(economics=economics)
    obs, _ = env.reset(options={"instance": instance})

    # Execute forward pass (no gradients needed)
    with torch.no_grad():
        result = policy.forward(obs, instance, economics, greedy=True)

    # Check outputs
    assert "x" in result
    assert "p" in result
    assert "vehicle_routes" in result
    assert "value" in result

    # Check output dimensions
    assert len(result["x"]) == n_regions
    assert len(result["p"]) == n_regions
    assert len(result["p"][0]) == n_regions
    assert len(result["vehicle_routes"]) == n_vehicles
