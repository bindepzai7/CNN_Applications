import os
import pandas as pd
from langid.langid import LanguageIdentifier, model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import re
import string
from task3_preprocess import preprocess_text, identify_vn, load_data, yield_tokens
from training import train, evaluate, save_results
from textCNN import TextCNN
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset
import json


folder_paths = {
    "train": "./data/ntc-scv/data/data_train/data_train/train",
    "valid": "./data/ntc-scv/data/data_train/data_train/test",
    "test": "./data/ntc-scv/data/data_test/data_test/test"
}

train_df = load_data(folder_paths["train"])
valid_df = load_data(folder_paths["valid"])
test_df = load_data(folder_paths["test"])

train_df_vi, train_df_other = identify_vn(train_df)

train_df_vi['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in train_df_vi.iterrows()]
valid_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in valid_df.iterrows()]
test_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in test_df.iterrows()]

tokenizer = get_tokenizer("basic_english")

vocab_size = 10000
vocabulary = build_vocab_from_iterator(
    yield_tokens(train_df_vi['preprocess_sentence'], tokenizer),
    max_tokens=vocab_size,
    specials=["<pad>", "<unk>"],
)
vocabulary.set_default_index(vocabulary["<unk>"])

def prepare_dataset(df):
    for idx, row in df.iterrows():
        sentence = row['preprocess_sentence']
        encoded_sentence = [vocabulary[token] for token in tokenizer(sentence)]
        label = row['label']
        yield encoded_sentence, label
        
train_dataset = prepare_dataset(train_df_vi)
train_dataset = to_map_style_dataset(train_dataset)

valid_dataset = prepare_dataset(valid_df)
valid_dataset = to_map_style_dataset(valid_dataset)

def collate_batch(batch):
    encoded_sentences, labels = [], []
    for encoded_sentence, label in batch:
        labels.append(label)
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)
        encoded_sentences.append(encoded_sentence)
        
    labels = torch.tensor(labels, dtype=torch.int64)
    encoded_sentences = pad_sequence(
        encoded_sentences,
        padding_value=vocabulary['<pad>'],
    )
    return encoded_sentences, labels

batch_size = 128
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)

def main():
    num_classes = 2
    vocab_size = len(vocabulary)
    embedding_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=num_classes
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    num_epochs = 10
    save_model = './model'
    os.makedirs(save_model, exist_ok=True)
    
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        train_acc, train_loss = train(model, optimizer, criterion, train_dataloader, device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        eval_acc, eval_loss = evaluate(model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)
        
        if eval_loss < best_loss_eval:
            best_loss_eval = eval_loss
            torch.save(model.state_dict(), os.path.join(save_model, 'model3.pt'))
        
        print("-" * 60)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")
        print("-" * 60)

    results_file = 'results/Sentiment Analysis/result.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    save_results(train_accs, train_losses, eval_accs, eval_losses, results_file)
    print(f"Results saved to {results_file}")
        
    model.load_state_dict(torch.load(os.path.join(save_model, 'model3.pt')))
    model.eval()
        
if __name__ == "__main__":
    main()