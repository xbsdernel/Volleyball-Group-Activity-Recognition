"""
Model 7: Two-Stage Hierarchical Deep Temporal Model (Full)
Paper: Ibrahim et al., CVPR 2016 — Section 3, Figure 2, Equations 7 & 8
Architecture:
Stage 1 (Person-level):
  - Person CNN (ResNet-50 backbone, replacing AlexNet per agreed substitution) -> 2048 dim
  - LSTM 1 (1024 hidden units) trained on individual player action labels
Stage 2 (Group-level):
  - Concatenation: P_tk = [x_tk ; h_tk] per player per frame (2048 + 1024 = 3072 dim) [Eq. 7]
  - Max-pool over K=12 players: Z_t = max_k (P_tk) -> (B, 9, 3072) [Eq. 8]
  - FC projection layer (3000 units) + ReLU + Dropout(0.5) [Section 3.3]
  - Group LSTM 2 (500 hidden units) [Section 3.3]
  - Linear classification head -> 8 group activity classes
"""

import torch
import torch.nn as nn
from torchvision import models


class TwoStageHierarchicalModel(nn.Module):
    """
    Full Two-Stage Hierarchical Deep Temporal Model for Group Activity Recognition.
    """
    def __init__(
        self,
        h1: int = 1024,
        proj: int = 3000,
        h2: int = 500,
        num_classes: int = 8,
        num_actions: int = 9,
        dropout: float = 0.5,
        pretrained_backbone: bool = True
    ):
        super().__init__()

        # Stage 1: Person-level components
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = models.resnet50(weights=weights)
        self.person_cnn = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)

        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=h1, num_layers=1, batch_first=True)
        self.person_action_head = nn.Linear(h1, num_actions)

        # Stage 2: Group-level components
        concat_dim = 2048 + h1  # x_tk concat h_tk = 3072
        self.fc_proj = nn.Sequential(
            nn.Linear(concat_dim, proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.lstm2 = nn.LSTM(input_size=proj, hidden_size=h2, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.group_classifier = nn.Linear(h2, num_classes)

    def forward_stage1(self, tracklet: torch.Tensor) -> torch.Tensor:
        """
        tracklet: (B, 9, 3, 224, 224) -> action logits: (B, 9, num_actions)
        """
        B, T, C, H, W = tracklet.shape
        feats = self.person_cnn(tracklet.view(B * T, C, H, W)).view(B, T, 2048)
        lstm1_out, _ = self.lstm1(feats)
        return self.person_action_head(lstm1_out)

    def forward_stage2(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 12, 9, 3, 224, 224) -> group logits: (B, num_classes)
        """
        B, K, T, C, H, W = x.shape
        x_flat = x.view(B * K, T, C, H, W)

        # 1. CNN features x_tk: (B*K, 9, 2048)
        with torch.no_grad():
            cnn_feats = self.person_cnn(x_flat.view(B * K * T, C, H, W)).view(B * K, T, 2048)

        # 2. LSTM 1 hidden states h_tk: (B*K, 9, h1)
        lstm1_out, _ = self.lstm1(cnn_feats)

        # 3. Concatenation: P_tk = [x_tk ; h_tk]  [Paper Eq. 7]
        p_tk = torch.cat([cnn_feats, lstm1_out], dim=-1)  # (B*K, 9, 2048 + h1)
        p_tk = p_tk.view(B, K, T, -1).permute(0, 2, 1, 3)  # (B, 9, 12, concat_dim)

        # 4. Max-pool over K=12 players: Z_t = max_k (P_tk)  [Paper Eq. 8]
        z_t = p_tk.max(dim=2)[0]  # (B, 9, concat_dim)

        # 5. FC projection (3000 units)
        z_proj = self.fc_proj(z_t)  # (B, 9, proj)

        # 6. Group LSTM 2 (500 units)
        _, (hn, _) = self.lstm2(z_proj)
        h_group = hn[-1]  # (B, h2)

        # 7. Group classification head
        return self.group_classifier(self.dropout(h_group))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stage2(x)
