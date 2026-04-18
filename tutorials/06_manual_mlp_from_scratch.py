import torch
import torch.nn.functional as F


def make_data(n: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    # Each point has 2 coordinates, so x shape is (n, 2).
    x = torch.rand(n, 2) * 2 - 1
    # Class 1 if x1 and x2 have the same sign, else class 0.
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    x += 0.15 * torch.randn_like(x)
    return x, y


def relu(x: torch.Tensor) -> torch.Tensor:
    # ReLU keeps positive values and clips negative values to 0.
    return torch.clamp(x, min=0.0)


def main() -> None:
    x, y = make_data()
    torch.manual_seed(42)

    # Raw parameters for a 2 -> 16 -> 2 network.
    # w1 maps 2 input features to 16 hidden features.
    w1 = (torch.randn(2, 16) * 0.5).requires_grad_()
    b1 = torch.zeros(16, requires_grad=True)
    # w2 maps the 16 hidden features to 2 class logits.
    w2 = (torch.randn(16, 2) * 0.5).requires_grad_()
    b2 = torch.zeros(2, requires_grad=True)
    params = [w1, b1, w2, b2]

    learning_rate = 0.05

    for epoch in range(150):
        # Manual forward pass.
        # x shape (600, 2) @ w1 shape (2, 16) -> (600, 16)
        hidden_pre_activation = x @ w1 + b1
        # ReLU does not change shape; it only changes values.
        hidden = relu(hidden_pre_activation)
        # (600, 16) @ (16, 2) -> (600, 2)
        logits = hidden @ w2 + b2

        # cross_entropy expects raw logits and integer class labels.
        loss = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            for param in params:
                param -= learning_rate * param.grad

        # Every parameter tensor has its own accumulated gradient.
        for param in params:
            param.grad.zero_()

        if epoch % 25 == 0 or epoch == 149:
            # argmax turns the 2 logits into the predicted class index.
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
