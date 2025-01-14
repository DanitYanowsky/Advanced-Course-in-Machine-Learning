import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

################### Complete the code below ###################
class Net(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.conv_layers= nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, 4, 4)
            # nn.ReLU(),
            # nn.Conv2d(128, 512, kernel_size=3, padding=1),  # (batch_size, 512, 4, 4)
            nn.ReLU(),
        )
        self.fc1=nn.Linear(128*4*4, 128*4)
        self.fc2=nn.Linear(128*4, 80)
        self.fc3=nn.Linear(80, latent_dim)

        self.relu=nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self,x):
        x=self.conv_layers(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = self.fc3(x)
        # x= self.softmax(x)
        return x
###############################################################

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# take a stratified subset of the training data, keeping only 5000 samples, with 500 samples per class
train_targets = train_dataset.targets
train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

################### Complete the code below ###################
num_classes = 10
model = Net(latent_dim =num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
################### Complete the code below ###################

# Training loop
val_losses = []
val_accuracies = []
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # moves the model to training mode
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        ################### Complete the code below ###################
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ###############################################################
        running_loss += loss.item()

    # Validation
    model.eval()  # moves the model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():  # Temporarily set all the requires_grad flags to false
        for images, labels in tqdm(test_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    val_loss /= len(test_loader)
    accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)
    print(f"accuracy: {accuracy}")

################### Complete the code below ###################
# Plot validation loss
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', marker='o')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='blue', marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
###############################################################