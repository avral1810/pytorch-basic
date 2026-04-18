import torch
import torch.nn.functional as F


VOCAB_SIZE = 15
SEQ_LEN = 10
EMBED_DIM = 24
HIDDEN_SIZE = 32
NUM_CLASSES = 2


def make_data(n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    tokens = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))
    first_half = tokens[:, : SEQ_LEN // 2].float().mean(dim=1)
    second_half = tokens[:, SEQ_LEN // 2 :].float().mean(dim=1)
    labels = (second_half > first_half).long()
    return tokens, labels


def main() -> None:
    train_x, train_y = make_data(800)
    val_x, val_y = make_data(200)
    torch.manual_seed(17)

    embedding_table = (torch.randn(VOCAB_SIZE, EMBED_DIM) * 0.1).requires_grad_()
    wxh = (torch.randn(EMBED_DIM, HIDDEN_SIZE) * 0.2).requires_grad_()
    whh = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.2).requires_grad_()
    bh = torch.zeros(HIDDEN_SIZE, requires_grad=True)
    why = (torch.randn(HIDDEN_SIZE, NUM_CLASSES) * 0.2).requires_grad_()
    by = torch.zeros(NUM_CLASSES, requires_grad=True)
    params = [embedding_table, wxh, whh, bh, why, by]

    learning_rate = 0.03
    batch_size = 64

    for epoch in range(20):
        permutation = torch.randperm(train_x.size(0))

        for start in range(0, train_x.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            tokens = train_x[indices]
            labels = train_y[indices]

            embedded = embedding_table[tokens]
            hidden = torch.zeros(tokens.size(0), HIDDEN_SIZE)

            for t in range(tokens.size(1)):
                x_t = embedded[:, t, :]
                hidden = torch.tanh(x_t @ wxh + hidden @ whh + bh)

            logits = hidden @ why + by
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            with torch.no_grad():
                for param in params:
                    param -= learning_rate * param.grad

            for param in params:
                param.grad.zero_()

        with torch.no_grad():
            embedded = embedding_table[val_x]
            hidden = torch.zeros(val_x.size(0), HIDDEN_SIZE)
            for t in range(val_x.size(1)):
                x_t = embedded[:, t, :]
                hidden = torch.tanh(x_t @ wxh + hidden @ whh + bh)

            logits = hidden @ why + by
            preds = logits.argmax(dim=1)
            acc = (preds == val_y).float().mean().item()
            val_loss = F.cross_entropy(logits, val_y).item()

        if epoch % 5 == 0 or epoch == 19:
            print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_acc={acc:.3f}")

    print("Focus on the explicit recurrent loop and how one hidden state is updated step by step.")


if __name__ == "__main__":
    main()
