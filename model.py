from torch import nn
from facenet_pytorch import InceptionResnetV1

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim=512, reduction=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.attn(x)
        return x * w, w


class FaceClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2")

        self.attention = SpatialAttention()

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        emb = self.backbone(x)
        emb, _ = self.attention(emb)
        return self.head(emb)