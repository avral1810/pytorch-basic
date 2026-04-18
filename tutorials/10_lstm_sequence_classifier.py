import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VOCAB_SIZE = 15
SEQ_LEN = 10


class LSTMClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Embedding converts integer token ids into dense vectors.
        self.embedding = nn.Embedding(VOCAB_SIZE, 32)
        # batch_first=True means shapes are (batch, seq_len, features).
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.classifier = nn.Linear(32, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: (batch, seq_len)
        embedded = self.embedding(tokens)
        # embedded shape: (batch, seq_len, 32)
        outputs, (hidden_state, cell_state) = self.lstm(embedded)

        # hidden_state shape is (num_layers, batch, hidden_size).
        # For one-layer LSTM, hidden_state[-1] is the last layer's summary.
        final_hidden = hidden_state[-1]
        return self.classifier(final_hidden)


def make_data(n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(21)
    tokens = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))

    # Label sequences by whether later tokens are larger on average than earlier ones.
    first_half = tokens[:, : SEQ_LEN // 2].float().mean(dim=1)
    second_half = tokens[:, SEQ_LEN // 2 :].float().mean(dim=1)
    labels = (second_half > first_half).long()
    return tokens, labels


def main() -> None:
    x, y = make_data()
    torch.manual_seed(21)
    # First 800 sequences for training, last 200 for validation.
    train_loader = DataLoader(TensorDataset(x[:800], y[:800]), batch_size=64, shuffle=True)
    val_x, val_y = x[800:], y[800:]

    model = LSTMClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(20):
        model.train()
        for tokens, labels in train_loader:
            # logits shape: (batch, 2)
            logits = model(tokens)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            # argmax chooses the class with the largest logit.
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()
            val_loss = loss_fn(val_logits, val_y).item()

        print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
