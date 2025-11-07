import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 

#  Extracted features from Xception

class FrozenXceptionClassifier(nn.Module):
    def __init__(self, out_feature, num_classes=3, pretrained=True):
        super(FrozenXceptionClassifier, self).__init__()
        
        self.interfea = nn.Linear(2048, out_feature)
        self.classifier = nn.Linear(out_feature, num_classes)

    def forward(self, x):
            # shape: [B, C]
        if x.dim() > 2:
            x = x.mean(dim=[2, 3])    # Global average pooling if needed
            
        features = self.interfea(x)
        logits = self.classifier(features)          # final output
        
        return features, logits