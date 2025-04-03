from torch import nn
import torch
import time

def train(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    # Counters for entire epoch
    total_acc, total_count = 0, 0
    # Counters for logging intervals
    running_acc, running_count = 0, 0
    losses = []
    start_time = time.time()
    
    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs)
        
        loss = criterion(predictions, labels)
        losses.append(loss.item())
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        # Calculate accuracy for this batch
        batch_correct = (predictions.argmax(1) == labels).sum().item()
        running_acc += batch_correct
        running_count += labels.size(0)
        total_acc += batch_correct
        total_count += labels.size(0)
        
        # Log at the specified interval
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | interval accuracy {:8.3f} | time {:5.2f}s".format(
                    epoch, idx, len(train_dataloader), running_acc / running_count, elapsed
                )
            )
            running_acc, running_count = 0, 0  # Reset only the interval counters
            start_time = time.time()
    
    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def evaluate(model, criterion, valid_dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []
    
    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            losses.append(loss.item())
            
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    
    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss