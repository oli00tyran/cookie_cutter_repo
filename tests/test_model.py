import pytest
import torch

from cookie_cutter_project.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_model(batch_size: int) -> None:
    """Test model with different batch sizes."""
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), but got {y.shape}"


def test_error_on_wrong_shape():
    """Test that model raises ValueError on wrong input shape."""
    model = MyAwesomeModel()
    
    # Test wrong number of dimensions (3D instead of 4D)
    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        model(torch.randn(1, 2, 3))
    
    # Test wrong image shape
    with pytest.raises(ValueError, match="Expected each sample to have shape \\[1, 28, 28\\]"):
        model(torch.randn(1, 1, 28, 29))