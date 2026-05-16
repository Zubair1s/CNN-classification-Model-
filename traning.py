import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------
# 1. Preprocessing
# -------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# -------------------
# 2. Load dataset (IMPORTANT: your structure)
# -------------------
dataset = datasets.ImageFolder(root="images", transform=transform)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Class mapping:", dataset.class_to_idx)
# cat = 0, dog = 1 (usually)

# -------------------
# 3. CNN Model
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

model = CNN()

# -------------------
# 4. Loss + Optimizer
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------
# 5. Training loop
# -------------------
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------
# 6. Save model
# -------------------
torch.save(model.state_dict(), "model.pth")

print("Model trained and saved!")