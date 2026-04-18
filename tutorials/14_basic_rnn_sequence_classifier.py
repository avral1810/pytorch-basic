import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VOCAB_SIZE = 15
SEQ_LEN = 10


class RNNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Convert integer token ids into learned dense vectors.
        self.embedding = nn.Embedding(VOCAB_SIZE, 32)
        # A vanilla RNN keeps only a hidden state, unlike an LSTM which also has a cell state.
        self.rnn = nn.RNN(input_size=32, hidden_size=32, batch_first=True, nonlinearity="tanh")
        self.classifier = nn.Linear(32, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: (batch, seq_len)
        embedded = self.embedding(tokens)
        # embedded shape: (batch, seq_len, 32)
        outputs, hidden_state = self.rnn(embedded)

        # hidden_state shape: (num_layers, batch, hidden_size)
        final_hidden = hidden_state[-1]
        return self.classifier(final_hidden)


def make_data(n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    tokens = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))

    # Label sequences by whether the later half has a larger average token value.
    first_half = tokens[:, : SEQ_LEN // 2].float().mean(dim=1)
    second_half = tokens[:, SEQ_LEN // 2 :].float().mean(dim=1)
    labels = (second_half > first_half).long()
    return tokens, labels


def main() -> None:
    x, y = make_data()
    torch.manual_seed(17)
    train_loader = DataLoader(TensorDataset(x[:800], y[:800]), batch_size=64, shuffle=True)
    val_x, val_y = x[800:], y[800:]

    model = RNNClassifier()
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
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()
            val_loss = loss_fn(val_logits, val_y).item()

        print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    print("Focus on embedding shapes, recurrent hidden state, and how RNNs differ from LSTMs.")


if __name__ == "__main__":
    main()
