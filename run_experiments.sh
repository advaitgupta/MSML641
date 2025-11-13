#!/bin.bash

mkdir -p results/logs
mkdir -p results/plots

echo "model,activation,optimizer,seq_len,clipping,test_accuracy,test_f1,avg_epoch_time_s" > results/metrics.csv

EPOCHS=5

echo "Starting evaluation"

TOTAL_RUNS=72
CURRENT_RUN=0
START_TIME=$(date +%s)

for model in "rnn" "lstm" "bilstm"; do
  for optim in "adam" "sgd" "rmsprop"; do
    for seq_len in 25 50 100; do
      if [ "$model" == "rnn" ]; then
        activations=("relu" "tanh" "sigmoid")
      else
        activations=("none")
      fi

      for act in "${activations[@]}"; do
        for clip in "true" "false"; do
          
          if [ "$model" == "rnn" ] && [ "$act" == "sigmoid" ]; then
            echo "SKIPPING: Config: $model | $act | $optim | $seq_len | $clip"
            echo "Reason: PyTorch nn.RNN does not support 'sigmoid' nonlinearity."
            continue
          fi

          CURRENT_RUN=$((CURRENT_RUN + 1))
          ELAPSED_TIME=$(( $(date +%s) - START_TIME ))
          AVG_TIME_PER_RUN=0
          
          if [ $CURRENT_RUN -gt 1 ]; then
              AVG_TIME_PER_RUN=$(( ELAPSED_TIME / (CURRENT_RUN - 1) ))
          fi
          
          REMAINING_RUNS=$(( TOTAL_RUNS - CURRENT_RUN ))
          ESTIMATED_REMAINING_TIME=$(( REMAINING_RUNS * AVG_TIME_PER_RUN ))
          EST_TIME_STR=$(date -u -d @${ESTIMATED_REMAINING_TIME} +%Hh%Mm%Ss)
          
          echo "RUN [$CURRENT_RUN / $TOTAL_RUNS] (Est. Time Left: $EST_TIME_STR)"
          echo "Config: $model | $act | $optim | $seq_len | $clip"
          
          python src/train.py \
            --model-type $model \
            --activation $act \
            --optimizer $optim \
            --seq-len $seq_len \
            --grad-clipping $clip \
            --epochs $EPOCHS

          if [ $? -ne 0 ]; then
              echo "ERROR during execution."
              exit 1
          fi

        done
      done
    done
  done
done

echo "ll experiments complete."
echo "Generating plots"

python src/evaluate.py

echo "Process finished. Check results/"
