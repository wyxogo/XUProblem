from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class VGG19OF10(nn.Module):
    '''VGG19 model of 10 classification with VGG19 weights
    Args:
        pretrained (bool): If True, uses VGG19_Weights.DEFAULT, otherwise uses None
    '''
    def __init__(self, pretrained: bool, numclass:int):
        super(VGG19OF10, self).__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.premodel = vgg19(weights=weights)
        self.last_classifier = nn.Linear(1000, numclass)
    def forward(self,x):
        x = self.premodel(x)
        x = self.last_classifier(x)
        return x

def build_model(arg):
    '''Build VGG19OF10 model'''
    model = VGG19OF10(arg.pretrained, arg.numclass)
    return model
    