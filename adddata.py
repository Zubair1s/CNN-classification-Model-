import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# -------------------
# same model structure
# -------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# -------------------
# load model
# -------------------
model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# -------------------
# preprocessing
# -------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# -------------------
# test image
# -------------------
img = Image.open("catt.jpg")
img = transform(img).unsqueeze(0)

# -------------------
# prediction
# -------------------
output = model(img)
prediction = torch.argmax(output)

if prediction.item() == 0:
    print("Cat 🐱")
else:
    print("Dog 🐶")