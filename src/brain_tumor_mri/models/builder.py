from brain_tumor_mri.models.cnn_small import SmallCNN
from brain_tumor_mri.models.resnet18 import build_resnet18
from brain_tumor_mri.models.cnn_small_v2 import SmallCNNv2


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True, img_size: int = 224):
    model_name = model_name.lower()

    if model_name == "cnn_small":
        return SmallCNN(img_size=img_size, num_classes=num_classes)

    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    
    if model_name == "cnn_small_v2":
        return SmallCNNv2(num_classes=num_classes)

    raise ValueError(f"Modèle inconnu : {model_name}")