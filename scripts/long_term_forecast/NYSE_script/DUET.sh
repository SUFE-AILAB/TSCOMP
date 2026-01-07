# from DUET github
# https://github.com/decisionintelligence/DUET/blob/main/scripts/multivariate_forecast/ILI_script/DUET.sh

model_name=DUET
batch_size=8
d_ff=1024
d_model=256
e_layers=2
factor=3
fc_dropout=0
k=2
learning_rate=0.001
lradj='type1'
n_heads=1
num_experts=2
patch_len=48
patience=5
hidden_size=256

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nyse/ \
  --data_path nyse.csv \
  --model_id nyse_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers $e_layers \
  --d_layers 1 \
  --d_model $d_model \
  --d_ff 512 \
  --hidden_size $hidden_size \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout 0.1 \
  --fc_dropout $fc_dropout \
  --moving_avg 25 \
  --lradj $lradj \
  --batch_size $batch_size \
  --learning_rate $learning_rate  \
  --train_epochs 100 \
  --loss 'MAE' \
  --patience $patience \
  --num_experts $num_experts \
  --noisy_gating \
  --k $k \
  --CI \
  --factor $factor \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nyse/ \
  --data_path nyse.csv \
  --model_id nyse_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size $hidden_size \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout 0.05 \
  --fc_dropout $fc_dropout \
  --moving_avg 25 \
  --lradj $lradj \
  --batch_size $batch_size \
  --learning_rate $learning_rate  \
  --train_epochs 100 \
  --loss 'MAE' \
  --patience $patience \
  --num_experts $num_experts \
  --noisy_gating \
  --k $k \
  --CI \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor $factor \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nyse/ \
  --data_path nyse.csv \
  --model_id nyse_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --d_model 128 \
  --d_ff $d_ff \
  --hidden_size $hidden_size \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout 0.15 \
  --fc_dropout $fc_dropout \
  --moving_avg 25 \
  --lradj $lradj \
  --batch_size $batch_size \
  --learning_rate 0.0005  \
  --train_epochs 100 \
  --loss 'MAE' \
  --patience $patience \
  --num_experts $num_experts \
  --noisy_gating \
  --k $k \
  --CI \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor $factor \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/nyse/ \
  --data_path nyse.csv \
  --model_id nyse_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size $hidden_size \
  --n_heads 2 \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout 0.05 \
  --fc_dropout $fc_dropout \
  --moving_avg 25 \
  --lradj $lradj \
  --batch_size $batch_size \
  --learning_rate 0.001  \
  --train_epochs 100 \
  --loss 'MAE' \
  --patience $patience \
  --num_experts $num_experts \
  --noisy_gating \
  --k 1 \
  --CI \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor $factor \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1