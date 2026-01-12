

model_name=RAFT

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_720_96 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-2 \
  --topm 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_720_192 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-2 \
  --topm 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_720_336 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-2 \
  --topm 20

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_720_720 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-2 \
  --topm 20