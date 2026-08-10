import importlib.util

import pytest


pytestmark = pytest.mark.contract


def test_required_public_torx_symbols_when_installed():
    if importlib.util.find_spec("torx") is None:
        pytest.skip("Torx optional dependency is not installed")

    from adze_t.torx_api.contracts import inspect_public_surface

    surface = inspect_public_surface()
    missing = [name for name, present in surface.items() if not present]
    assert not missing, f"Missing public Torx symbols: {missing}"
