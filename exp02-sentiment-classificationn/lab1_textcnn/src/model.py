import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, num_classes=2, kernel_sizes=[2,3,4,5], num_channels=200, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0,2,1)   # (batch, embed_dim, seq_len)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2) for c in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
