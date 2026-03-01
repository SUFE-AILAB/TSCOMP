

model_name=RAFT

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_96 \
  --model $model_name \
  --data custom \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --seq_len 720 \
  --pred_len 96 \
  --learning_rate 1e-2 \
  --topm 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_192 \
  --model $model_name \
  --data custom \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --seq_len 720 \
  --pred_len 192 \
  --learning_rate 1e-3 \
  --topm 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_336 \
  --model $model_name \
  --data custom \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --seq_len 720 \
  --pred_len 336 \
  --learning_rate 1e-3 \
  --topm 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_720 \
  --model $model_name \
  --data custom \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --seq_len 720 \
  --pred_len 720 \
  --learning_rate 1e-3 \
  --topm 1