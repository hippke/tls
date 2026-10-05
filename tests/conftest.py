import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # golden_cases


def pytest_collection_modifyitems(config, items):
    """Network tests run only with TLS_NETWORK_TESTS=1."""
    if os.environ.get("TLS_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="needs network; set TLS_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
