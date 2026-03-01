

model_name=RAFT

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720_96 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-2 \
  --topm 10


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720_192 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-3 \
  --topm 10

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720_336 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-3 \
  --topm 20

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720_720 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-4 \
  --topm 20