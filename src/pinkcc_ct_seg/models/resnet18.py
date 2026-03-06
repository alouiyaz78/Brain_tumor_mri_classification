from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 pour classification binaire via 2 logits.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Gèle tout le backbone, ne laisse entraînable que la tête FC.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_layer4_and_fc(model: nn.Module) -> nn.Module:
    """
    Dé-gèle layer4 + fc pour un fine-tuning léger.
    """
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    return model