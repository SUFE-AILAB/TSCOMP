
model_name=TimeLLM

for pred_len in 96 192 336 720
do
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 24 \
  --d_model 32 \
  --d_ff 128 \
  --learning_rate 0.01 \
  --llm_layers 6 \
  --train_epochs 100 \
  --few_shot_ratio 0.05 \
  --des 'Exp' \
  --itr 1
done
