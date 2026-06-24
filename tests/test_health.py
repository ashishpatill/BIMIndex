import sys
import os

# Ensure the project root is on the path so 'core_processor' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_processor.health import check


def test_check_returns_expected_dict():
    result = check()
    assert result == {"status": "ok", "version": "1.0"}
