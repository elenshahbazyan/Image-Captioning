import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """ResNet-50 encoder that outputs image feature vectors."""
    def __init__(self, embed_size=256, train_cnn=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # remove final FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.train_cnn = train_cnn

        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad_(False)

    def forward(self, images):
        with torch.set_grad_enabled(self.train_cnn):
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
