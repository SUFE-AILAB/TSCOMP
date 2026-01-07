

model_name=RAFT

python3 -u run.py \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_720_96 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 96 \
  --learning_rate 1e-4 \
  --topm 1


python3 -u run.py \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_720_192 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 192 \
  --learning_rate 1e-3 \
  --topm 1

python3 -u run.py \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_720_336 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 336 \
  --learning_rate 1e-3 \
  --topm 10

python3 -u run.py \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_720_720 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 720 \
  --learning_rate 1e-4 \
  --topm 20