# 09_efficientnet_b0.ipynb sera créé via Jupyter
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_efficientnet_b0(num_classes=2, pretrained=True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


