import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LogisticRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # One linear layer is enough for logistic regression.
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCEWithLogitsLoss expects raw logits, so no sigmoid here.
        return self.linear(x)


def make_data(n: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(12)
    # 400 examples, each with 2 features.
    x = torch.randn(n, 2)
    scores = 2.0 * x[:, 0] - 1.2 * x[:, 1] + 0.3 * torch.randn(n)
    y = (scores > 0).float().unsqueeze(1)
    return x, y


def main() -> None:
    x, y = make_data()
    torch.manual_seed(12)
    # DataLoader lets us train with smaller chunks of data instead of all 400 at once.
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    model = LogisticRegression()
    # BCEWithLogitsLoss combines sigmoid + binary cross entropy internally.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(120):
        for batch_x, batch_y in loader:
            # logits shape is (batch_size, 1)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 119:
            with torch.no_grad():
                # During evaluation we turn logits into probabilities for human interpretation.
                probs = torch.sigmoid(model(x))
                preds = (probs >= 0.5).float()
                acc = (preds == y).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
