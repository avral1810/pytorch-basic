import torch
import torch.nn.functional as F


IMAGE_SIZE = 16
NUM_CLASSES = 3


def generate_pattern(label: int, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    image = torch.zeros(image_size, image_size)

    if label == 0:
        image[:, image_size // 2 - 1:image_size // 2 + 1] = 1.0
    elif label == 1:
        image[image_size // 2 - 1:image_size // 2 + 1, :] = 1.0
    else:
        idx = torch.arange(image_size)
        image[idx, idx] = 1.0

    image += 0.10 * torch.randn_like(image)
    return image.clamp(0.0, 1.0).unsqueeze(0)


def make_dataset(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    torch.manual_seed(123)

    for i in range(size):
        label = i % NUM_CLASSES
        images.append(generate_pattern(label))
        labels.append(label)

    return torch.stack(images), torch.tensor(labels)


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def main() -> None:
    train_x, train_y = make_dataset(900)
    test_x, test_y = make_dataset(180)
    torch.manual_seed(123)

    conv1_weight = (torch.randn(8, 1, 3, 3) * 0.1).requires_grad_()
    conv1_bias = torch.zeros(8, requires_grad=True)
    conv2_weight = (torch.randn(16, 8, 3, 3) * 0.1).requires_grad_()
    conv2_bias = torch.zeros(16, requires_grad=True)
    fc1_weight = (torch.randn(16 * 4 * 4, 32) * 0.1).requires_grad_()
    fc1_bias = torch.zeros(32, requires_grad=True)
    fc2_weight = (torch.randn(32, NUM_CLASSES) * 0.1).requires_grad_()
    fc2_bias = torch.zeros(NUM_CLASSES, requires_grad=True)
    params = [
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    ]

    learning_rate = 0.03
    batch_size = 64

    for epoch in range(20):
        permutation = torch.randperm(train_x.size(0))

        for start in range(0, train_x.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            images = train_x[indices]
            labels = train_y[indices]

            x = F.conv2d(images, conv1_weight, conv1_bias, padding=1)
            x = relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = F.conv2d(x, conv2_weight, conv2_bias, padding=1)
            x = relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = x.flatten(start_dim=1)
            x = relu(x @ fc1_weight + fc1_bias)
            logits = x @ fc2_weight + fc2_bias

            loss = F.cross_entropy(logits, labels)
            loss.backward()

            with torch.no_grad():
                for param in params:
                    param -= learning_rate * param.grad

            for param in params:
                param.grad.zero_()

        with torch.no_grad():
            x = F.conv2d(test_x, conv1_weight, conv1_bias, padding=1)
            x = relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = F.conv2d(x, conv2_weight, conv2_bias, padding=1)
            x = relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = x.flatten(start_dim=1)
            x = relu(x @ fc1_weight + fc1_bias)
            logits = x @ fc2_weight + fc2_bias
            preds = logits.argmax(dim=1)
            acc = (preds == test_y).float().mean().item()
            test_loss = F.cross_entropy(logits, test_y).item()

        if epoch % 5 == 0 or epoch == 19:
            print(f"epoch={epoch:02d} test_loss={test_loss:.4f} test_acc={acc:.3f}")

    print("Focus on raw conv weights, pooling, flattening, and the final manual classifier.")


if __name__ == "__main__":
    main()
