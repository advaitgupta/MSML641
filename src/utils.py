import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def set_seeds(seed_val: int = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_pred_logits, y_true):
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred_labels = torch.round(y_pred_probs)
    
    y_true_cpu = y_true.cpu().numpy()
    y_pred_labels_cpu = y_pred_labels.cpu().detach().numpy()
    
    accuracy = accuracy_score(y_true_cpu, y_pred_labels_cpu)
    f1 = f1_score(y_true_cpu, y_pred_labels_cpu, average='macro')
    
    return accuracy, f1

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs