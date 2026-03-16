"""
Shared pytest fixtures for PhaseForget-Zettel tests.
"""

import os
import sys
import tempfile

import pytest

# Ensure src is on the Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
