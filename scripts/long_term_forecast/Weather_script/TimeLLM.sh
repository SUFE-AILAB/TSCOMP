
model_name=TimeLLM

for pred_len in 96 192 336 720
do
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 24 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --llm_layers 6 \
  --train_epochs 10 \
  --few_shot_ratio 0.05 \
  --des 'Exp' \
  --itr 1
done
