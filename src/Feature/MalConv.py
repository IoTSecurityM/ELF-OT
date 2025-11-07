import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 


class MalConv(nn.Module):
    def __init__(self, input_length=300000,  # e.g., 1MB
                 emb_dim=8,
                 num_classes=3,
                 kernel_size=500,
                 stride=500):
        super(MalConv, self).__init__()

        self.input_length = input_length
        self.embed = nn.Embedding(257, emb_dim, padding_idx=256)  # 0-255 bytes + padding

        self.conv1 = nn.Conv1d(emb_dim, 128, kernel_size=kernel_size, stride=stride)
        self.conv1_gate = nn.Conv1d(emb_dim, 128, kernel_size=kernel_size, stride=stride)

        self.pooling = nn.AdaptiveMaxPool1d(1)  # Global max pooling

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch_size, input_length) as byte integers [0, 255]
        x = self.embed(x)  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L) for Conv1d

        # Gated convolution
        x_feature = self.conv1(x)
        x_gate = torch.sigmoid(self.conv1_gate(x))
        x = x_feature * x_gate  # gated output

        # Global max pooling
        x = self.pooling(x).squeeze(-1)  # (B, 128)

        # Fully connected output
        out = self.fc(x)  # (B, num_classes)
        return x, out
    