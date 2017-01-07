#!/bin/sh
Data='BBN'
for ((i=1; i<=5; i++)); do
  ### Train AFET
  echo 'Step '$i
  echo 'Train AFET'
  Model/pl_warp $Data 50 0.01 50 10 0.25 3 1 1 5 1
  echo ' '

  ### Predict and evaluate
  echo 'Evalation AFET'
  source /shared/installation/p2_env/bin/activate
  python Evaluation/emb_prediction.py $Data pl_warp bipartite maximum cosine 0.12
  deactivate
  source /shared/installation/p3_env/bin/activate
  mkdir -p final_results/AFET_$Data.$i
  python fair_evaluation.py $Data final_results/AFET_$Data.$i/data_log.csv
  deactivate
done
