from brain_tumor_mri.models.cnn_small import SmallCNN
from brain_tumor_mri.models.resnet18 import build_resnet18
from brain_tumor_mri.models.cnn_small_v2 import SmallCNNv2

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True, img_size: int = 224):
    model_name = model_name.lower()

    if model_name == "cnn_small":
        return SmallCNN(img_size=img_size, num_classes=num_classes)

    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)

    if model_name == "cnn_small_v2":
        return SmallCNNv2(num_classes=num_classes)

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Modèle inconnu : {model_name}")