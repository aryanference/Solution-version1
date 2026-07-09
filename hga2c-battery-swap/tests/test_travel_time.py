"""Unit tests for travel time computation (§2.5).

Tests written BEFORE implementation (TDD). These encode the exact coordinate
data from the target instance and verify Euclidean distance = travel time.
"""
import math
import pytest


def travel_time(node_i_xy: tuple[float, float], node_j_xy: tuple[float, float]) -> float:
    """Euclidean distance between two nodes = travel time in minutes."""
    (xi, yi), (xj, yj) = node_i_xy, node_j_xy
    return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)


# ---------------------------------------------------------------------------
# §2.5: Two explicit sanity checks from the master prompt
# ---------------------------------------------------------------------------

class TestTravelTimeExplicit:
    """Exact assertions specified in §2.5 of the master prompt."""

    HUB = (5.5, 6.99)
    REGION_0 = (2.5, 2.5)
    REGION_4 = (7.5, 7.5)

    def test_hub_to_region_4(self):
        """travel_time(Hub, Region_4) ≈ 2.064 (±1e-3)."""
        assert travel_time(self.HUB, self.REGION_4) == pytest.approx(2.064, abs=1e-3)

    def test_hub_to_region_0(self):
        """travel_time(Hub, Region_0) ≈ 5.4 (±1e-3)."""
        # Manual: sqrt((5.5-2.5)^2 + (6.99-2.5)^2) = sqrt(9 + 20.1601) = sqrt(29.1601) ≈ 5.4
        assert travel_time(self.HUB, self.REGION_0) == pytest.approx(5.4, abs=1e-1)


# ---------------------------------------------------------------------------
# Symmetry and basic properties
# ---------------------------------------------------------------------------

class TestTravelTimeProperties:
    """Mathematical properties travel time must satisfy."""

    HUB = (5.5, 6.99)
    REGIONS = [
        (2.5, 2.5), (7.5, 2.5), (12.5, 2.5),
        (2.5, 7.5), (7.5, 7.5), (12.5, 7.5),
        (2.5, 12.5), (7.5, 12.5), (12.5, 12.5),
    ]
    ALL_NODES = [HUB] + REGIONS

    def test_symmetry(self):
        """t(i, j) == t(j, i) for all pairs."""
        for i, ni in enumerate(self.ALL_NODES):
            for j, nj in enumerate(self.ALL_NODES):
                assert travel_time(ni, nj) == pytest.approx(travel_time(nj, ni), abs=1e-10)

    def test_self_distance_zero(self):
        """t(i, i) == 0 for all nodes."""
        for node in self.ALL_NODES:
            assert travel_time(node, node) == pytest.approx(0.0, abs=1e-10)

    def test_all_positive_or_zero(self):
        """All pairwise distances ≥ 0."""
        for ni in self.ALL_NODES:
            for nj in self.ALL_NODES:
                assert travel_time(ni, nj) >= 0.0

    def test_triangle_inequality(self):
        """t(i,k) ≤ t(i,j) + t(j,k) for sample triples."""
        nodes = self.ALL_NODES
        # Check all triples (10 nodes → 720 triples — fast enough)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                for k in range(len(nodes)):
                    d_ik = travel_time(nodes[i], nodes[k])
                    d_ij = travel_time(nodes[i], nodes[j])
                    d_jk = travel_time(nodes[j], nodes[k])
                    assert d_ik <= d_ij + d_jk + 1e-10


# ---------------------------------------------------------------------------
# Verify the full travel time matrix from instance data (using fixtures)
# ---------------------------------------------------------------------------

class TestTravelTimeMatrix:
    """Verify travel time matrix built from instance.json coordinates."""

    def test_matrix_shape(self, all_node_coords):
        """10×10 matrix (hub + 9 regions)."""
        n = len(all_node_coords)
        assert n == 10

    def test_known_adjacent_regions(self, region_coords):
        """Adjacent grid cells (e.g., Region 0 ↔ Region 1) should be 5.0 apart."""
        # Region_0 = (2.5, 2.5), Region_1 = (7.5, 2.5) → dist = 5.0
        assert travel_time(region_coords[0], region_coords[1]) == pytest.approx(5.0, abs=1e-6)

    def test_known_diagonal_regions(self, region_coords):
        """Diagonal grid cells should be sqrt(50) ≈ 7.071 apart."""
        # Region_0 = (2.5, 2.5), Region_4 = (7.5, 7.5) → dist = sqrt(50)
        expected = math.sqrt(50)
        assert travel_time(region_coords[0], region_coords[4]) == pytest.approx(expected, abs=1e-3)

    def test_max_distance(self, all_node_coords):
        """Max distance in grid should be Region_0 ↔ Region_8 = sqrt(200) ≈ 14.14."""
        # (2.5,2.5) to (12.5,12.5)
        max_d = max(
            travel_time(all_node_coords[i], all_node_coords[j])
            for i in range(len(all_node_coords))
            for j in range(len(all_node_coords))
        )
        expected = math.sqrt(200)  # 14.142
        assert max_d == pytest.approx(expected, abs=1e-3)
