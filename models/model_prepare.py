import torch.nn as nn
from torchvision import models
import logging
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)

# TODO: make freeze
# Depend on feature_extract, close the requires_grad fo model param
def set_parameter_requires_grad(model, feature_extracting):
    # feature_extracting==True -> only train last layer -> set all model parameters no grad
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# create the model architecture from TorchVision
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    preModel = None
    if model_name == "resnet18":
        preModel = models.resnet18(pretrained=use_pretrained)
        preModel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        set_parameter_requires_grad(preModel, feature_extract)
        # model_ft.relu = nn.LeakyReLU() #[Experiment Phase]
        num_features = preModel.fc.in_features
        preModel.fc = nn.Linear(num_features, num_classes)  # Add Fully-Connected Layer
        # input = 224, ftrs = 512
    elif model_name == "resnet50":
        preModel = models.resnet50(pretrained=use_pretrained)
        preModel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        set_parameter_requires_grad(preModel, feature_extract)
        num_features = preModel.fc.in_features
        preModel.fc = nn.Linear(num_features, num_classes)  # Add Fully-Connected Layer
        # input 224, ftrs = 2048
    elif model_name == "efficientNet_B0":
        if use_pretrained:
            preModel = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            preModel = EfficientNet.from_name('efficientnet-b0')
        # preModel._conv_stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # set_parameter_requires_grad(preModel, feature_extract)
        num_features = preModel._fc.in_features
        preModel._fc = nn.Linear(num_features, num_classes)
    elif model_name == "densenet":
        preModel = models.densenet121(pretrained=use_pretrained)
        preModel.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        set_parameter_requires_grad(preModel, feature_extract)
        num_ftrs = preModel.classifier.in_features
        preModel.classifier = nn.Linear(num_ftrs, num_classes)

    return preModel


if __name__ == "__main__":
    model = initialize_model("densenet", num_classes=2, feature_extract=False, use_pretrained=True)
    # model._fc.in_features = 2
    # print(model._conv_stem)
    print(model)