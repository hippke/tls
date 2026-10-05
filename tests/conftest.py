import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # golden_cases

# The repo's regression tests assert reference values of the exact statistic.
# Approximate backends (e.g. the default "fused-binned") are covered by
# test_backends_equivalence.py. Override with TLS_BACKEND=... if desired.
os.environ.setdefault("TLS_BACKEND", "fused")


def pytest_collection_modifyitems(config, items):
    """Network tests run only with TLS_NETWORK_TESTS=1."""
    if os.environ.get("TLS_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="needs network; set TLS_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
