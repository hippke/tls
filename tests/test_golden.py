"""Characterization test: complete TLS results must match the frozen golden set.
See golden_cases.py. Regenerate (only for intended, documented numerical changes):
    python tests/golden_cases.py --write
"""

import pytest
from golden_cases import CASES, compare, run


@pytest.mark.parametrize("name", list(CASES))
def test_golden(name):
    bad = compare(name, run(name))
    assert not bad, f"{name}: {bad}"
