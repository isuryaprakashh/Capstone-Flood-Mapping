"""
Pytest configuration and shared fixtures for SpaceNet 8 testing.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_batch():
    """Create a sample synthetic batch for testing models and losses."""
    B, C, H, W = 2, 3, 128, 128
    num_classes = 5
    pre = torch.rand(B, C, H, W, dtype=torch.float32)
    post = torch.rand(B, C, H, W, dtype=torch.float32)
    mask = (torch.rand(B, num_classes, H, W) > 0.8).float()
    return {
        "pre_image": pre,
        "post_image": post,
        "mask": mask,
    }


@pytest.fixture
def synthetic_images():
    """Create a synthetic numpy image pair."""
    H, W = 256, 256
    pre = np.random.uniform(0, 1, (3, H, W)).astype(np.float32)
    post = np.random.uniform(0, 1, (3, H, W)).astype(np.float32)
    return pre, post


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)
