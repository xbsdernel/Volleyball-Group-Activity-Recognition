"""
Baseline 7: Two-Stage Model Without LSTM 2
Paper: Ibrahim et al., CVPR 2016 — Section 4.1, Baseline 7
Stage 1: Person CNN (ResNet-50 backbone) + Person LSTM 1 (1024 hidden units).
Stage 2: B7Model:
- Frozen Person CNN + LSTM 1
- Max-pooling last hidden state over all 12 players: group_repr = max_k (h_K) -> (B, 1024)
- NO Group LSTM 2!
- Group classification head: Dropout(0.5) -> Linear(1024, 8)
"""

import torch
import torch.nn as nn
from torchvision import models


class PersonLSTM1(nn.Module):
    """
    Stage 1: Person-level temporal model.
    CNN Backbone + LSTM 1 (1024 units) predicting individual player action classes.
    """
    def __init__(self, num_actions: int = 9, hidden_dim: int = 1024, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward_features(self, tracklet: torch.Tensor):
        """
        tracklet: (B, 9, 3, 224, 224) -> lstm1_out: (B, 9, hidden_dim), last_h: (B, hidden_dim)
        """
        B, T, C, H, W = tracklet.shape
        feats = self.backbone(tracklet.view(B * T, C, H, W)).view(B, T, 2048)
        lstm_out, (hn, _) = self.lstm1(feats)
        return lstm_out, hn[-1]

    def forward(self, tracklet: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.forward_features(tracklet)
        return self.action_head(self.dropout(lstm_out))


class B7Model(nn.Module):
    """
    Stage 2: Baseline 7 Group Activity Model (Two-stage model WITHOUT LSTM 2).
    Max-pools the last hidden states of LSTM 1 over all 12 players and feeds directly
    into a linear classifier.
    """
    def __init__(
        self,
        person_lstm1: PersonLSTM1 = None,
        num_classes: int = 8,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        freeze_stage1: bool = True
    ):
        super().__init__()
        if person_lstm1 is not None:
            self.person_lstm1 = person_lstm1
        else:
            self.person_lstm1 = PersonLSTM1(hidden_dim=hidden_dim, dropout=dropout, pretrained=True)

        if freeze_stage1:
            for p in self.person_lstm1.parameters():
                p.requires_grad = False  # Frozen Stage 1

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 9, 3, 224, 224)
        B, K, T, C, H, W = x.shape
        x_flat = x.view(B * K, T, C, H, W)
        with torch.no_grad():
            _, last_h = self.person_lstm1.forward_features(x_flat)  # (B*K, hidden_dim)

        # Reshape to (B, K, hidden_dim) and max-pool over players
        last_h = last_h.view(B, K, -1)   # (B, 12, hidden_dim)
        group_repr = last_h.max(dim=1)[0]  # (B, hidden_dim) — max over 12 players
        return self.classifier(self.dropout(group_repr))
