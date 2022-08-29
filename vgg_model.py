from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class VGG19TO10(nn.Module):
    def __init__(self, pretrained: bool, numclass:int):
        super(VGG19TO10, self).__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.premodel = vgg19(weights=weights)
        self.last_classifier = nn.Linear(1000, numclass)
    def forward(self,x):
        x = self.premodel(x)
        x = self.last_classifier(x)
        return x

def build_model(arg):
    model = VGG19TO10(arg.pretrained, arg.numclass)
    return model