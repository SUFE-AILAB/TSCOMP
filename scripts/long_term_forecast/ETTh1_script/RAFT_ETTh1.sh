

model_name=RAFT

python3 -u run.py \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 720 \
  --pred_len 96 \
  --learning_rate 1e-3 \
  --topm 20


python3 -u run.py \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_192 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 720 \
  --pred_len 192 \
  --learning_rate 1e-2 \
  --topm 20

python3 -u run.py \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_336 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 720 \
  --pred_len 336 \
  --learning_rate 1e-2 \
  --topm 20

python3 -u run.py \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_720 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 720 \
  --pred_len 720 \
  --learning_rate 1e-4 \
  --topm 20