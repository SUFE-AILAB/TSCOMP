

model_name=RAFT

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_720_96 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --learning_rate 1e-2 \
  --topm 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_720_192 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --learning_rate 1e-3 \
  --topm 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_720_336 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --learning_rate 1e-3 \
  --topm 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_720_720 \
  --model $model_name \
  --data custom \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --learning_rate 1e-3 \
  --topm 1