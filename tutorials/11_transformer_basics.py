import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VOCAB_SIZE = 20
SEQ_LEN = 8
EMBED_DIM = 32
NUM_CLASSES = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()

        # Transformers do not know token order by themselves,
        # so we add a position-dependent signal to each embedding.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TinyTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Embedding maps token ids to dense vectors.
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoding = PositionalEncoding(EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # We pool the sequence into one vector and classify it.
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: (batch, sequence_length)
        x = self.embedding(tokens)
        x = self.pos_encoding(x)
        x = self.encoder(x)

        # Mean pooling is a simple way to turn a sequence into one vector.
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def make_sequence_data(n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(99)
    tokens = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))

    # Label 1 if the sequence has a large total token value, else label 0.
    labels = (tokens.sum(dim=1) > (SEQ_LEN * VOCAB_SIZE / 2)).long()
    return tokens, labels


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def main() -> None:
    x, y = make_sequence_data()
    torch.manual_seed(99)
    split = 800

    # Use the last 200 examples as validation.
    train_loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=64, shuffle=True)
    val_x, val_y = x[split:], y[split:]

    model = TinyTransformerClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(20):
        model.train()
        for tokens, labels in train_loader:
            logits = model(tokens)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_acc = accuracy(val_logits, val_y)
            val_loss = loss_fn(val_logits, val_y).item()
        print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    print("Focus on embeddings, sequence shape, encoder layers, and pooling.")


if __name__ == "__main__":
    main()
