from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 pour classification binaire via 2 logits.
    Si le téléchargement des poids échoue, on retombe sur une initialisation aléatoire.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    try:
        model = models.resnet18(weights=weights)
        if pretrained:
            print("✅ ResNet18 chargé avec poids préentraînés.")
    except Exception as e:
        print(f"⚠️ Impossible de télécharger les poids préentraînés : {e}")
        print("➡️ Fallback sur ResNet18 sans poids préentraînés.")
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_layer4_and_fc(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    return model