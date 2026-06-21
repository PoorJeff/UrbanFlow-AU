from importlib.metadata import version

import urbanflow


def test_package_exposes_installed_version() -> None:
    assert urbanflow.__version__ == version("urbanflow-au")
