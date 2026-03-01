

model_name=RAFT

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_96_24 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 24 \
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
  --model_id ili_96_36 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 36 \
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
  --model_id ili_96_48 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 48 \
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
  --model_id ili_96_60 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --pred_len 60 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 1e-2 \
  --topm 20