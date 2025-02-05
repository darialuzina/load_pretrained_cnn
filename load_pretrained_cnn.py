import torch.nn as nn
import torchvision.models as models


def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool=True):
    if model_name == 'alexnet' :
        model = models.alexnet(pretrained = pretrained)
        model.classifier[6] = nn.Linear(in_features = 4096, out_features=num_classes)
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained = pretrained)
        model.classifier[6] = nn.Linear(in_features = 4096, out_features=num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained = pretrained)
        model.fc = nn.Linear(in_features = 1024, out_features=num_classes)
        model.aux1.fc1 = nn.Linear(in_features=2048, out_features=num_classes)
        model.aux1.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        model.aux2.fc1 = nn.Linear(in_features=2048, out_features=num_classes)
        model.aux2.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(in_features = 512, out_features=num_classes)
    
    return model