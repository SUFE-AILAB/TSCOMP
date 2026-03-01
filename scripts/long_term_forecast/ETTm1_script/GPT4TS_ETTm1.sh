
seq_len=512
model=GPT4TS

for pred_len in 96 192 336 720
do

python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --few_shot_ratio 0.05 \
    --data_path ETTm1.csv \
    --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
    --data ETTm1 \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --patch_len 16 \
    --stride 16 \
    --llm_layers 6 \
    --itr 1 \
    --model $model \
    --is_gpt 1
done