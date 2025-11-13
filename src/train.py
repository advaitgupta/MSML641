import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import pandas as pd
from tqdm import tqdm 

from models import SentimentClassifier
from preprocess import get_dataloaders
from utils import set_seeds, calculate_metrics, epoch_time

set_seeds(42)

def get_optimizer(model, optimizer_name, lr=1e-3):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer")

def train(model, iterator, optimizer, criterion, grad_clipping, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    pbar = tqdm(iterator, desc=f'Epoch {epoch+1:02} Train', unit='batch', leave=False)

    for labels, text in pbar:
        labels, text = labels.to(device), text.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(text)
        
        loss = criterion(predictions, labels)
        acc, f1 = calculate_metrics(predictions.squeeze(), labels.squeeze())
        
        loss.backward()
        
        if grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_f1 += f1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    pbar = tqdm(iterator, desc='Evaluate', unit='batch', leave=False)

    with torch.no_grad():
        for labels, text in pbar:
            labels, text = labels.to(device), text.to(device)
            
            predictions = model(text)
            
            loss = criterion(predictions, labels)
            acc, f1 = calculate_metrics(predictions.squeeze(), labels.squeeze())
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Analysis Training')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['rnn', 'lstm', 'bilstm'])
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh', 'sigmoid', 'none'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--seq-len', type=int, default=50, choices=[25, 50, 100])
    parser.add_argument('--grad-clipping', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    config_name = f"{args.model_type}_{args.activation}_{args.optimizer}_{args.seq_len}_{'clip' if args.grad_clipping else 'noclip'}"
    print(f"Starting Experiment: {config_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    BATCH_SIZE = 32
    EMBED_DIM = 100
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.4 

    print(f"Loading data for seq_len={args.seq_len}...")
    train_loader, test_loader, vocab_size = get_dataloaders(args.seq_len, BATCH_SIZE)
    print(f"Vocab size: {vocab_size}")

    model = SentimentClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        model_type=args.model_type,
        activation=args.activation,
        bidirectional=(args.model_type == 'bilstm')
    ).to(device)
    
    optimizer = get_optimizer(model, args.optimizer)
    
    criterion = nn.BCEWithLogitsLoss().to(device) 

    epoch_logs = []
    total_epoch_time = 0
    best_test_f1 = -1

    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, args.grad_clipping, epoch)
        
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
        
        end_time = time.time()
        total_epoch_time += (end_time - start_time)
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.3f}')
        print(f'\tTest  Loss: {test_loss:.3f} | Test  Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.3f}')

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1
        }
        epoch_logs.append(epoch_data)

    log_df = pd.DataFrame(epoch_logs)
    log_df.to_csv(f'results/logs/{config_name}.csv', index=False)
    
    final_metrics = {
        'model': args.model_type,
        'activation': args.activation,
        'optimizer': args.optimizer,
        'seq_len': args.seq_len,
        'clipping': args.grad_clipping,
        'test_accuracy': epoch_logs[-1]['test_acc'],
        'test_f1': epoch_logs[-1]['test_f1'],
        'avg_epoch_time_s': total_epoch_time / args.epochs
    }
    
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv('results/metrics.csv', mode='a', header=False, index=False)
    
    print(f"Finished Experiment: {config_name}")