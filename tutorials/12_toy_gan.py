import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The generator maps random noise vectors into fake 1D samples.
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The discriminator maps a 1D sample to one logit:
        # positive means "looks real", negative means "looks fake".
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_real(batch_size: int) -> torch.Tensor:
    # Real data comes from a simple 1D Gaussian centered near 2.
    return torch.randn(batch_size, 1) * 0.7 + 2.0


def sample_noise(batch_size: int) -> torch.Tensor:
    # Noise is the generator's starting point.
    return torch.randn(batch_size, 3)


def main() -> None:
    torch.manual_seed(0)
    generator = Generator()
    discriminator = Discriminator()

    loss_fn = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.003)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.003)

    for step in range(300):
        batch_size = 64

        # Train discriminator: classify real as 1 and fake as 0.
        real_samples = sample_real(batch_size)
        # detach() prevents generator gradients from being updated in the discriminator step.
        fake_samples = generator(sample_noise(batch_size)).detach()

        real_logits = discriminator(real_samples)
        fake_logits = discriminator(fake_samples)

        # The discriminator's targets are:
        # 1 for real samples, 0 for fake samples.
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)

        d_loss = loss_fn(real_logits, real_targets) + loss_fn(fake_logits, fake_targets)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train generator: fool the discriminator into predicting 1 for fake samples.
        generated = generator(sample_noise(batch_size))
        fooled_logits = discriminator(generated)
        # The generator wants fake samples to be judged as real.
        g_loss = loss_fn(fooled_logits, torch.ones_like(fooled_logits))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if step % 50 == 0 or step == 299:
            with torch.no_grad():
                samples = generator(sample_noise(1000))
                # fake_mean is a simple sanity check for whether generated samples
                # are moving toward the real-data region around 2.0.
                print(
                    f"step={step:03d} d_loss={d_loss.item():.4f} "
                    f"g_loss={g_loss.item():.4f} fake_mean={samples.mean().item():.3f}"
                )


if __name__ == "__main__":
    main()
