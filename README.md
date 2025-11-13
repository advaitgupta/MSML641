## 1. Setup Instructions

* **Python Version:** Python 3.8 or newer.
* **Dependencies:** All required packages are listed in `requirements.txt`.

1.  Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 2. How to Run Training and Evaluation

I have automated the entire thing in a single script including training everything and generating plots and everything.

### Step 1: Run All Experiments
Execute the `run_experiments.sh` script. This will iterate through all 72 valid combinations, train each model, and save the results.

```bash
bash run_experiments.sh
```
    

### Step 2: Generate Report
This step is handled **automatically** by the script from Step 1. After all experiments are finished, `src/evaluate.py` is called to generate the final plots and summary tables.

If you wish to re-generate the analysis *without* re-running training, you can run:

```bash
python src/evaluate.py
```

## 3. Expected Runtime and Output Files

* **Expected Runtime:** The full `run_experiments.sh` script will run 72 separate training experiments. This will take a significant amount of time, varying based on GPU hardware (from ~1-2 hours on an A100 GPU to much longer on older hardware).

* **Output Files:** After the script completes, the following files will be generated:
    * **`results/metrics.csv`**: A CSV file containing the final performance (Accuracy, F1, Time) for all 72 experiments.
    * **`results/plots/acc_f1_vs_seq_len.png`**: A plot comparing model performance across different sequence lengths.
    * **`results/plots/loss_vs_epochs.png`**: A plot comparing the training loss for the best and worst-performing models.
