from pathlib import Path

import matplotlib

# Must be set before pyplot is imported anywhere, so tests never open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from helpers import make_vessel_mask  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """The plotting functions create figures; don't let them pile up."""
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def data_dir():
    """The example images/masks shipped with the repo."""
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def healthy_mask():
    """Straight horizontal vessel with constant width (no stenosis)."""
    return make_vessel_mask(centers=())


@pytest.fixture
def stenotic_mask():
    """Straight vessel with one sharp narrowing at x = 170 (severity ~0.64)."""
    return make_vessel_mask(centers=(150,))
