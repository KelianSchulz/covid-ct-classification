import torch
import torch.nn as nn
from torchvision import models



def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model


if __name__ == "__main__":
    model = create_model()
    test_input = torch.randn(4, 3, 224, 224)
    test_output = model(test_input)
    print(test_output.shape)