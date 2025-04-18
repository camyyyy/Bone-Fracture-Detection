# alz_model.py
import os
import PIL
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class_names = ['Alzheimer Disease', 'Mild Alzheimer Risk', 'Moderate Alzheimer Risk',
               'Very Mild Alzheimer Risk', 'No Risk', 'Parkinson Disease']


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(class_names)).to(device)

    weights_path = os.path.abspath("C:\\Users\\Remotlab\\Bone-Fracture-Detection\\weights\\Vbai-DPA 2.0.pt")  # adapte si le fichier est ailleurs
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def predict(model, image_input, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image_input, PIL.Image.Image):
        image = image_input.convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()], probs[0][predicted.item()].item() * 100
