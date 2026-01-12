import os.path

import pytest
import torch

from cookie_cutter_project.data import corrupt_mnist, normalize, preprocess_data


def test_normalize():
    """Test that normalize function correctly normalizes images to mean=0, std=1."""
    # Create a tensor with known mean and std
    images = torch.randn(100, 1, 28, 28) * 5 + 10  # mean ~10, std ~5
    
    normalized = normalize(images)
    
    # Check that the output has approximately mean=0 and std=1
    assert normalized.mean().abs() < 0.1, f"Normalized images should have mean ~0, got {normalized.mean()}"
    assert (normalized.std() - 1.0).abs() < 0.1, f"Normalized images should have std ~1, got {normalized.std()}"


@pytest.mark.skipif(not os.path.exists("data/raw/train_images_0.pt"), reason="Raw data files not found")
def test_preprocess_data():
    """Test that preprocess_data loads and processes raw data correctly."""
    preprocess_data(raw_dir="data/raw", processed_dir="data/processed")
    
    # Check that processed files were created
    assert os.path.exists("data/processed/train_images.pt"), "Processed train images not found"
    assert os.path.exists("data/processed/train_target.pt"), "Processed train targets not found"
    assert os.path.exists("data/processed/test_images.pt"), "Processed test images not found"
    assert os.path.exists("data/processed/test_target.pt"), "Processed test targets not found"
    
    # Load and verify shapes
    train_images = torch.load("data/processed/train_images.pt", weights_only=True)
    train_targets = torch.load("data/processed/train_target.pt", weights_only=True)
    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_targets = torch.load("data/processed/test_target.pt", weights_only=True)
    
    assert train_images.shape[0] == train_targets.shape[0], "Train images and targets count mismatch"
    assert test_images.shape[0] == test_targets.shape[0], "Test images and targets count mismatch"
    assert train_images.dim() == 4, f"Train images should be 4D, got {train_images.dim()}D"


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, f"Train dataset should have 30000 samples, but got {len(train)}"
    assert len(test) == 5000, f"Test dataset should have 5000 samples, but got {len(test)}"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), f"Image should have shape (1, 28, 28), but got {x.shape}"
            assert y in range(10), f"Label should be in range 0-9, but got {y}"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Train set should contain all 10 digit classes"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "Test set should contain all 10 digit classes"