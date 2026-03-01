
model_name=Timer

seq_len=672
label_len=576
patch_len=96

# Loop over pred_len to match framework standard
for pred_len in 96 192 336 720
do
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 8 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 2048 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --itr 1 \
  \
  --few_shot_ratio 0.05 \
  --des 'Exp' 
done
