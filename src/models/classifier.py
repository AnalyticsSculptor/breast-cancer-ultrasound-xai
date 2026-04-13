import torch
import torch.nn as nn
import timm

class MCDropout(nn.Dropout):
    """Behaves as training mode even during evaluation for uncertainty estimation."""
    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=True, inplace=self.inplace)

class BreastCancerClassifier(nn.Module):
    def __init__(self, backbone_name: str = 'tf_efficientnet_b4.ns_jft_in1k', pretrained: bool = True, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features # 1792
        
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            MCDropout(p=dropout_rate),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            MCDropout(p=dropout_rate),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True