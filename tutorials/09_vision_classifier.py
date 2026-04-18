import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 16
NUM_CLASSES = 3


def generate_pattern(label: int, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    # Build a simple synthetic image so we can practice vision pipelines
    # without downloading a real dataset yet.
    image = torch.zeros(image_size, image_size)

    if label == 0:
        image[:, image_size // 2 - 1:image_size // 2 + 1] = 1.0
    elif label == 1:
        image[image_size // 2 - 1:image_size // 2 + 1, :] = 1.0
    else:
        idx = torch.arange(image_size)
        image[idx, idx] = 1.0

    image += 0.10 * torch.randn_like(image)

    # CNNs expect a channel dimension, so shape becomes (1, H, W).
    return image.clamp(0.0, 1.0).unsqueeze(0)


class PatternDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.images = []
        self.labels = []
        torch.manual_seed(123)

        for i in range(size):
            # Cycle through the classes so the dataset stays balanced.
            label = i % NUM_CLASSES
            self.images.append(generate_pattern(label))
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.labels[index]


class VisionCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # A small CNN is enough for this toy vision problem.
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main() -> None:
    torch.manual_seed(123)
    train_loader = DataLoader(PatternDataset(900), batch_size=64, shuffle=True)
    test_loader = DataLoader(PatternDataset(180), batch_size=64)

    model = VisionCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(15):
        model.train()

        for images, labels in train_loader:
            # The model sees a batch of images and returns class scores.
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        print(f"epoch={epoch:02d} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

    print("This script teaches the vision pipeline without needing a dataset download.")


if __name__ == "__main__":
    main()
