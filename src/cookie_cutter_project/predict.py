import random

import matplotlib.pyplot as plt
import torch
import typer
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def predict(
    model_checkpoint: str = "models/model.pth",
    num_images: int = 9,
    save_path: str = "reports/figures/predictions.png",
) -> None:
    """Show random test images with model predictions."""
    # Load model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Load test data
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    # Select random indices
    indices = random.sample(range(len(test_images)), num_images)

    # Calculate grid size
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    with torch.inference_mode():
        for i, idx in enumerate(indices):
            image = test_images[idx].unsqueeze(0).to(DEVICE)
            true_label = test_target[idx].item()

            # Get prediction
            output = model(image)
            pred_label = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item() * 100

            # Plot image
            ax = axes[i]
            ax.imshow(test_images[idx].squeeze(), cmap="gray")
            
            # Color title based on correct/incorrect
            color = "green" if pred_label == true_label else "red"
            ax.set_title(f"Pred: {pred_label} (True: {true_label})\n{confidence:.1f}% confidence", color=color)
            ax.axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    typer.run(predict)
