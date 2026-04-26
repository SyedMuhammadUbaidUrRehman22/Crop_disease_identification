from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


ModelName = Literal[
    "mobilenetv2",
    "resnet18",
    "efficientnet_b0",
    "resnet50",
    "efficientnet_b3",
    "densenet201",
]


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all model parameters.

    This is useful when using transfer learning and training only the final
    classification head first.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreezes all model parameters.

    This is useful for full fine-tuning after the classification head has
    been trained.
    """
    for param in model.parameters():
        param.requires_grad = True


def build_mobilenetv2(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    Builds MobileNetV2 with a custom classification head.

    MobileNetV2 is used as a lightweight baseline model.
    """

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_resnet18(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    Builds ResNet18 with a custom classification head.

    ResNet18 is used as a standard CNN baseline.
    """

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_efficientnet_b0(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    Builds EfficientNet-B0 with a custom classification head.

    EfficientNet-B0 is included because it aligns with the selected paper
    and is computationally efficient.
    """

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_resnet50(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    Builds ResNet50 with a custom classification head.

    ResNet50 is one of the architectures evaluated in the original paper.
    """

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_efficientnet_b3(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    Builds EfficientNet-B3 with a custom classification head.

    EfficientNet-B3 is the strongest architecture reported in the selected
    paper, but it is heavier than EfficientNet-B0.
    """

    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_densenet201(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    Builds DenseNet201 with a custom classification head.

    DenseNet201 is one of the architectures evaluated in the original paper.
    """

    weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet201(weights=weights)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def create_model(
    model_name: ModelName,
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_features: bool = False
) -> nn.Module:
    """
    Creates a CNN model for plant disease classification.

    Args:
        model_name: Name of the architecture.
        num_classes: Number of output disease classes.
        dropout: Dropout probability in the classification head.
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_features: Whether to freeze the feature extractor.

    Returns:
        A PyTorch neural network model.
    """

    if model_name == "mobilenetv2":
        model = build_mobilenetv2(num_classes, dropout, pretrained)

    elif model_name == "resnet18":
        model = build_resnet18(num_classes, dropout, pretrained)

    elif model_name == "efficientnet_b0":
        model = build_efficientnet_b0(num_classes, dropout, pretrained)

    elif model_name == "resnet50":
        model = build_resnet50(num_classes, dropout, pretrained)

    elif model_name == "efficientnet_b3":
        model = build_efficientnet_b3(num_classes, dropout, pretrained)

    elif model_name == "densenet201":
        model = build_densenet201(num_classes, dropout, pretrained)

    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            "Choose from: mobilenetv2, resnet18, efficientnet_b0, "
            "resnet50, efficientnet_b3, densenet201."
        )

    if freeze_features:
        freeze_backbone(model)

        # Re-enable classifier parameters after freezing backbone
        if model_name in ["mobilenetv2", "efficientnet_b0", "efficientnet_b3"]:
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif model_name in ["resnet18", "resnet50"]:
            for param in model.fc.parameters():
                param.requires_grad = True

        elif model_name == "densenet201":
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Counts trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """
    Counts all parameters in a model.
    """
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    NUM_CLASSES = 16

    model = create_model(
        model_name="efficientnet_b0",
        num_classes=NUM_CLASSES,
        dropout=0.3,
        pretrained=True,
        freeze_features=False
    )

    print(model)
    print(f"Total parameters: {count_total_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")