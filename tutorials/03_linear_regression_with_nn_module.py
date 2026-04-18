import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # A linear layer implements y_hat = xW^T + b.
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)

    # unsqueeze(1) turns shape (200,) into shape (200, 1),
    # which is what nn.Linear(1, 1) expects for one input feature.
    x = torch.linspace(-2, 2, steps=200).unsqueeze(1)
    noise = 0.25 * torch.randn_like(x)
    y = 3.5 * x + 1.2 + noise
    return x, y


def main() -> None:
    x, y = make_data()
    torch.manual_seed(7)
    # TensorDataset pairs each x row with its matching y row.
    # DataLoader then groups them into batches.
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    model = LinearRegressionModel()
    # MSELoss is the standard regression loss.
    loss_fn = nn.MSELoss()
    # SGD will update model.parameters() using the gradients from backward().
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(100):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            # batch_x shape is usually (32, 1) except possibly the last batch.
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)

            # Standard PyTorch training recipe:
            # 1. clear old gradients
            # 2. compute new gradients
            # 3. apply the update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 99:
            print(f"epoch={epoch:03d} loss={total_loss / len(loader):.4f}")

    print("learned weight:", model.linear.weight.item())
    print("learned bias:", model.linear.bias.item())
    print("This should be close to the true line y = 3.5x + 1.2")


if __name__ == "__main__":
    main()
