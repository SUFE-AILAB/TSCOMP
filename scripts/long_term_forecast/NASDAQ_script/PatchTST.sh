

model_name=PatchTST

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nasdaq/ \
  --data_path nasdaq.csv \
  --model_id nasdaq_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nasdaq/ \
  --data_path nasdaq.csv \
  --model_id nasdaq_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nasdaq/ \
  --data_path nasdaq.csv \
  --model_id nasdaq_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nasdaq/ \
  --data_path nasdaq.csv \
  --model_id nasdaq_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --n_heads 16 \
  --itr 1