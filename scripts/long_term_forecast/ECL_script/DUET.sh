model_name=DUET
batch_size=16
d_ff=1024
d_model=512
dropout=0.5
e_layers=4
factor=3
fc_dropout=0
k=2
learning_rate=0.0005
lradj='type1'
n_heads=1
num_experts=4
patch_len=48
patience=5

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 512 \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout $dropout \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 512 \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout $dropout \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 512 \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout $dropout \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 512 \
  --n_heads $n_heads \
  --seg_len 6 \
  --win_size 2\
  --activation 'gelu' \
  --patch_len $patch_len \
  --stride 8 \
  --period_len 4 \
  --dropout $dropout \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1