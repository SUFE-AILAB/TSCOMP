model_name=DUET
batch_size=32
d_ff=512
dropout=0.2
e_layers=2
factor=3
fc_dropout=0.5
learning_rate=0.0005
lradj='type3'
n_heads=8
num_experts=4
patience=5
k=2
patch_len=48
d_model=512

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 256 \
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
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 256 \
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
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 256 \
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
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --d_model $d_model \
  --d_ff $d_ff \
  --hidden_size 256 \
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
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1