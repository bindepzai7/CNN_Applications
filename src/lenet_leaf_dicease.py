import os 
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import torchsummary as summary

from training import train, evaluate, save_results
import time

data_dir = './data/cassavaleafdata'
data_path = {
    'train': os.path.join(data_dir, 'train'),
    'valid': os.path.join(data_dir, 'validation'),
    'test': os.path.join(data_dir, 'test')
}

def loader(path):
    return Image.open(path)

img_size = 150
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(root=data_path['train'], loader=loader, transform=train_transforms)
valid_data = datasets.ImageFolder(root=data_path['valid'], transform=train_transforms)
test_data = datasets.ImageFolder(root=data_path['test'], transform=train_transforms)

BATCH_SIZE = 64
train_dataloader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

class LeNetClassifier2(nn.Module):
    def __init__(self, num_classes):
        super(LeNetClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 35 * 35, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def main():
    num_classes = len(train_data.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNetClassifier2(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = 30
    save_model = './model'
    os.makedirs(save_model, exist_ok=True)
    
    train_accs = []
    train_losses = []
    eval_accs = []
    eval_losses = []
    best_loss_eval = 100
    
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        
        train_acc, train_loss = train(model, optimizer, criterion, train_dataloader, device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        eval_acc, eval_loss = evaluate(model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)
        
        if eval_loss < best_loss_eval:
            best_loss_eval = eval_loss
            torch.save(model.state_dict(), os.path.join(save_model, 'model2.pt'))
            
        print("-"*60)
        print(f"Epoch {epoch} | Time: {time.time() - epoch_start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")
        print("-"*60)
        
    # Save results to JSON file
    results_file = 'results/Cassava Leaf Disease/result.json'
    save_results(train_accs, train_losses, eval_accs, eval_losses, results_file)
    print(f"Results saved to {results_file}")
    
    # Plotting the training and validation losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Training Loss', color='blue')
    plt.plot(range(num_epochs), eval_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 2)
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label='Training Accuracy', color='blue')
    plt.plot(range(num_epochs), eval_accs, label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig('Plots/Cassava Leaf Disease.png')
    plt.show()
    
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNetClassifier2(num_classes=5).to(device)
    model.load_state_dict(torch.load(os.path.join('./model', 'model2.pt')))
    
    test_dataloader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    test_acc, test_loss = evaluate(model, criterion, test_dataloader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
if __name__ == "__main__":
    main()