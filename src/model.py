import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, filter_size=3,
                 num_filters=100, dropout=0.5, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(filter_size, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        emb = self.embedding(x)               # (batch, L, D)
        emb = emb.unsqueeze(1)                # (batch, 1, L, D)
        c = self.conv(emb)                    # (batch, num_filters, L-k+1, 1)
        c = torch.relu(c.squeeze(3))          # (batch, num_filters, L-k+1)
        pooled = torch.max(c, dim=2)[0]       # (batch, num_filters)
        out = self.fc(self.dropout(pooled))   # (batch, 2)
        return out
