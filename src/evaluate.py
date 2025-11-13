import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.csv")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

def generate_summary_table(df):
    print("\nComparative Analysis Summary Table")
    df_sorted = df.sort_values(by='test_f1', ascending=False)
    print(df_sorted.to_string()) 
    
    print("\nBest Performing Configuration")
    print(df_sorted.iloc[0])

def plot_acc_f1_vs_seq_len(df):
    print("Generating plot: Accuracy/F1 vs. Sequence Length (Best Performers)")
    
    best_indices = df.groupby('seq_len')['test_f1'].idxmax()
    
    df_plot = df.loc[best_indices].copy()

    print("\nBest model for each sequence length")
    print(df_plot[['model', 'optimizer', 'seq_len', 'test_f1']].to_string())

    df_melted = df_plot.melt(
        id_vars=['seq_len'], 
        value_vars=['test_accuracy', 'test_f1'], 
        var_name='metric', 
        value_name='score'
    )
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='seq_len', y='score', hue='metric', style='metric', markers=True, markersize=10)
    
    plt.title('Best Accuracy/F1 vs. Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Score')
    plt.grid(True)
    plt.xticks([25, 50, 100])
    plt.savefig(os.path.join(PLOTS_DIR, 'acc_f1_vs_seq_len.png'))
    plt.close()

def get_log_filename(run_series):
    model = run_series['model']
    
    if model in ['lstm', 'bilstm']:
        activation = 'none'
    else:
        activation = run_series['activation']
    
    optimizer = run_series['optimizer']
    seq_len = int(run_series['seq_len'])
    clipping = 'clip' if run_series['clipping'] else 'noclip'
    
    return f"{model}_{activation}_{optimizer}_{seq_len}_{clipping}"

def plot_loss_curves(df):
    print("Generating plot: Training Loss vs. Epochs")
    df_sorted = df.sort_values(by='test_f1', ascending=True)
    
    worst_run = df_sorted.iloc[0]
    best_run = df_sorted.iloc[-1]
    
    worst_name = get_log_filename(worst_run)
    best_name = get_log_filename(best_run)

    try:
        df_worst = pd.read_csv(os.path.join(LOGS_DIR, f"{worst_name}.csv"))
        df_best = pd.read_csv(os.path.join(LOGS_DIR, f"{best_name}.csv"))
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_worst['epoch'], df_worst['train_loss'], label=f'Worst: {worst_name}')
        plt.plot(df_best['epoch'], df_best['train_loss'], label=f'Best: {best_name}')
        
        plt.title('Training Loss vs. Epochs (Best vs. Worst Model)')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, 'loss_vs_epochs.png'))
        plt.close()
        
    except FileNotFoundError as e:
        print(f"Error: Could not find log file for plotting. {e}")


if __name__ == "__main__":
    if not os.path.exists(METRICS_FILE):
        print(f"Error: {METRICS_FILE} not found. Run 'bash run_experiments.sh' first.")
        exit(1)
        
    df = pd.read_csv(METRICS_FILE)
    
    df['activation'] = df.apply(
        lambda row: 'n/a' if row['model'] in ['lstm', 'bilstm'] else row['activation'],
        axis=1
    )
    
    generate_summary_table(df)
    plot_acc_f1_vs_seq_len(df)
    plot_loss_curves(df)
    
    print(f"\nEvaluation complete. Plots saved to {PLOTS_DIR}")