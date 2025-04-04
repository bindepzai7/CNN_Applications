import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from matplotlib import pyplot as plt
from PIL import Image

from training import train, evaluate, save_results
import time


ROOT = './data'
train_data = datasets.MNIST(root=ROOT, train=True, download=True)
test_data = datasets.MNIST(root=ROOT, train=False, download=True)

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

mean = train_data.dataset.data.float().mean() / 255.0
std = train_data.dataset.data.float().std() / 255.0

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_data.dataset.transform = train_transform
valid_data.dataset.transform = valid_transform

BATCH_SIZE = 64
train_dataloader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

class LeNetClassifier1(nn.Module):
    def __init__(self, num_classes):
        super(LeNetClassifier1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc1(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc3(outputs)
        return outputs
        
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    num_classes = len(train_data.dataset.classes)
    
    lenet_model = LeNetClassifier1(num_classes=num_classes).to(device)
    optimizer = optim.Adam(lenet_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 10
    save_model = './model'
    os.makedirs(save_model, exist_ok=True)
    
    #tracking
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        #train
        train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        #validate
        eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)
        
        if eval_loss < best_loss_eval:
            best_loss_eval = eval_loss
            torch.save(lenet_model.state_dict(), os.path.join(save_model, 'model1.pt'))
            
        print("-" * 59)
        print("| End of epoch {:3d} | Time: {:5.2f}s | Train Acc: {:8.3f} | Train Loss: {:8.3f} |".format(
            epoch, time.time() - epoch_start_time, train_acc, train_loss))
        print("| Valid Acc: {:8.3f} | Valid Loss: {:8.3f}".format(eval_acc, eval_loss))
        print("-" * 59)
    
    lenet_model.load_state_dict(torch.load(os.path.join(save_model, 'model1.pt')))
    lenet_model.eval()
    save_model = './results/Digit Recognition'
    file_name = os.path.join(save_model, 'result.json')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    #save results to a JSON file
    save_results(train_accs, train_losses, eval_accs, eval_losses, file_name)
    print("Best model loaded.")
    
if __name__ == "__main__":
    main()
        
    