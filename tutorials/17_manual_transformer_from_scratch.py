import math

import torch
import torch.nn.functional as F


VOCAB_SIZE = 20
SEQ_LEN = 8
EMBED_DIM = 32
NUM_CLASSES = 2


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def make_data(n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(99)
    tokens = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))
    labels = (tokens.sum(dim=1) > (SEQ_LEN * VOCAB_SIZE / 2)).long()
    return tokens, labels


def main() -> None:
    train_x, train_y = make_data(800)
    val_x, val_y = make_data(200)
    torch.manual_seed(99)

    embedding_table = (torch.randn(VOCAB_SIZE, EMBED_DIM) * 0.1).requires_grad_()
    w_q = (torch.randn(EMBED_DIM, EMBED_DIM) * 0.2).requires_grad_()
    w_k = (torch.randn(EMBED_DIM, EMBED_DIM) * 0.2).requires_grad_()
    w_v = (torch.randn(EMBED_DIM, EMBED_DIM) * 0.2).requires_grad_()
    w_o = (torch.randn(EMBED_DIM, EMBED_DIM) * 0.2).requires_grad_()
    w_ff1 = (torch.randn(EMBED_DIM, 64) * 0.2).requires_grad_()
    b_ff1 = torch.zeros(64, requires_grad=True)
    w_ff2 = (torch.randn(64, EMBED_DIM) * 0.2).requires_grad_()
    b_ff2 = torch.zeros(EMBED_DIM, requires_grad=True)
    w_cls = (torch.randn(EMBED_DIM, NUM_CLASSES) * 0.2).requires_grad_()
    b_cls = torch.zeros(NUM_CLASSES, requires_grad=True)
    params = [embedding_table, w_q, w_k, w_v, w_o, w_ff1, b_ff1, w_ff2, b_ff2, w_cls, b_cls]

    pe = positional_encoding(SEQ_LEN, EMBED_DIM)
    learning_rate = 0.02
    batch_size = 64
    scale = math.sqrt(EMBED_DIM)

    for epoch in range(20):
        permutation = torch.randperm(train_x.size(0))

        for start in range(0, train_x.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            tokens = train_x[indices]
            labels = train_y[indices]

            x = embedding_table[tokens] + pe.unsqueeze(0)
            q = x @ w_q
            k = x @ w_k
            v = x @ w_v

            attention_scores = q @ k.transpose(1, 2) / scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attended = attention_weights @ v
            x = attended @ w_o

            ff_hidden = torch.relu(x @ w_ff1 + b_ff1)
            x = ff_hidden @ w_ff2 + b_ff2

            pooled = x.mean(dim=1)
            logits = pooled @ w_cls + b_cls
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            with torch.no_grad():
                for param in params:
                    param -= learning_rate * param.grad

            for param in params:
                param.grad.zero_()

        with torch.no_grad():
            x = embedding_table[val_x] + pe.unsqueeze(0)
            q = x @ w_q
            k = x @ w_k
            v = x @ w_v
            attention_scores = q @ k.transpose(1, 2) / scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attended = attention_weights @ v
            x = attended @ w_o
            ff_hidden = torch.relu(x @ w_ff1 + b_ff1)
            x = ff_hidden @ w_ff2 + b_ff2
            pooled = x.mean(dim=1)
            logits = pooled @ w_cls + b_cls
            preds = logits.argmax(dim=1)
            acc = (preds == val_y).float().mean().item()
            val_loss = F.cross_entropy(logits, val_y).item()

        if epoch % 5 == 0 or epoch == 19:
            print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_acc={acc:.3f}")

    print("Focus on manual attention weights, sequence mixing, and mean pooling.")


if __name__ == "__main__":
    main()
