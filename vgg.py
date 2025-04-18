import os
import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import SSD

num_classes = 7

def create_model(num_classes):
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    num_default_boxes = len(model.anchor_generator.aspect_ratios[0]) * len(model.anchor_generator.scales)

    model.classification_headers = nn.ModuleList([
        nn.Conv2d(256, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(1024, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, num_classes * num_default_boxes, kernel_size=3, padding=1)
    ])

    model.regression_headers = nn.ModuleList([
        nn.Conv2d(256, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(1024, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, 4 * num_default_boxes, kernel_size=3, padding=1)
    ])

    custom_layers = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

    model.classifier = custom_layers
    model.to('cpu')
    return model

def get_vgg_model():
    model = create_model(num_classes=num_classes)

    # Correction du chemin avec affichage absolu
    model_path = os.path.abspath("C:\\Users\\Remotlab\\Bone-Fracture-Detection\\weights\\model_vgg.pt")
    print("===> Loading VGG model from:", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier du modèle VGG est introuvable : {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    return model
