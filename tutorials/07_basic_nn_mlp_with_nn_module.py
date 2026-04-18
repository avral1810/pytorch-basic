import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # This is a small multilayer perceptron:
        # input -> hidden -> hidden -> class scores
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_xor_like_data(n: int = 800) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    # Random 2D points in the square [-1, 1] x [-1, 1].
    x = torch.rand(n, 2) * 2 - 1

    # Points in quadrants I and III belong to class 1.
    # Points in quadrants II and IV belong to class 0.
    y = ((x[:, 0] * x[:, 1]) > 0).long()

    # Add noise so the decision boundary is not perfectly clean.
    x += 0.15 * torch.randn_like(x)
    return x, y


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # CrossEntropyLoss expects raw logits, so we convert them to class ids here.
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main() -> None:
    x, y = make_xor_like_data()
    torch.manual_seed(42)
    split = int(0.8 * len(x))

    # Use the first 80% for training and the rest for validation.
    train_ds = TensorDataset(x[:split], y[:split])
    val_x, val_y = x[split:], y[split:]

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = MLPClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(80):
        model.train()
        for batch_x, batch_y in train_loader:
            # The network returns two numbers per example:
            # one score for class 0 and one score for class 1.
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 79:
            # Switch to evaluation mode before validation.
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss = loss_fn(val_logits, val_y).item()
                val_acc = accuracy(val_logits, val_y)
            print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
