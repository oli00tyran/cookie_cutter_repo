import os.path

import pytest
import torch

from cookie_cutter_project.data import corrupt_mnist
from cookie_cutter_project.model import MyAwesomeModel


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_training_loop():
    """Test that one training step decreases the loss."""
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    # Get a single batch
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)
    img, target = next(iter(train_dataloader))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initial forward pass
    model.train()
    output = model(img)
    loss_before = loss_fn(output, target).item()

    # Training step
    optimizer.zero_grad()
    loss = loss_fn(model(img), target)
    loss.backward()
    optimizer.step()

    # After one step, loss should decrease (or at least change)
    output_after = model(img)
    loss_after = loss_fn(output_after, target).item()

    assert loss_after < loss_before, "Loss should decrease after one training step"


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_model_parameters_update():
    """Test that model parameters are updated after training step."""
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)
    img, target = next(iter(train_dataloader))

    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training step
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(img), target)
    loss.backward()
    optimizer.step()

    # Check that at least some parameters changed
    params_changed = False
    for p_initial, p_after in zip(initial_params, model.parameters()):
        if not torch.equal(p_initial, p_after):
            params_changed = True
            break

    assert params_changed, "Model parameters should be updated after training step"
