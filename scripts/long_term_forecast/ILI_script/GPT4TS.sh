
seq_len=104
model=GPT4TS


for pred_len in 24 36 48 60
do

python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --few_shot_ratio 0.05 \
    --data_path national_illness.csv \
    --model_id ili_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --stride 2 \
    --itr 1 \
    --model $model \
    --is_gpt 1
done