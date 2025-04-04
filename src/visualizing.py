import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

file_name = 'results/Sentiment Analysis/result.json'

train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Load the data from the JSON file
with open(file_name, 'r') as f:
    data = json.load(f)
    train_losses = data['train_losses']
    train_accs = data['train_accs']
    val_losses = data['eval_losses']
    val_accs = data['eval_accs']
    epoch = len(train_losses)
    
# Plotting the training and validation losses
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_losses, label='Training Loss', color='blue')
plt.plot(range(epoch), val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_accs, label='Training Accuracy', color='blue')
plt.plot(range(epoch), val_accs, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.savefig('Plots/SentimentAnalysis.png')
plt.show()